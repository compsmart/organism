from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .config import AgentConfig, TrainingConfig
from .env import Action, ObsIndex


class MemoryBuffer:
    """Sliding window of recent hidden states for episodic memory."""

    def __init__(self, capacity: int, hidden_size: int) -> None:
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.buffer: list[Tensor] = []

    def push(self, hidden: Tensor) -> None:
        self.buffer.append(hidden.detach())
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_tensor(self, device: torch.device) -> Tensor:
        if not self.buffer:
            return torch.zeros(1, 0, self.hidden_size, device=device)
        return torch.stack(self.buffer, dim=1)

    def clear(self) -> None:
        self.buffer.clear()


class EpisodicMemory(nn.Module):
    """Attention-based recall over recent hidden states."""

    def __init__(self, hidden_size: int, memory_slots: int = 16) -> None:
        super().__init__()
        self.memory_slots = memory_slots
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self._scale = hidden_size ** -0.5

    def forward(self, current_hidden: Tensor, memory_buffer: Tensor) -> Tensor:
        if memory_buffer.size(1) == 0:
            return current_hidden
        query = self.query_proj(current_hidden).unsqueeze(1)
        keys = self.key_proj(memory_buffer)
        scores = (query * keys).sum(-1) * self._scale
        weights = torch.softmax(scores, dim=-1)
        readout = (weights.unsqueeze(-1) * memory_buffer).sum(1)
        gate = self.output_gate(torch.cat([current_hidden, readout], dim=-1))
        return current_hidden + gate * readout


WORKSPACE_CHANNELS = {
    "food": [0, 1, 2, 11],        # food_left, food_center, food_right, food_contact
    "danger": [3, 4, 5, 6, 7, 8], # hazard L/C/R, wall L/C/R
    "shelter": [9, 10, 12],        # shelter_alignment, shelter_proximity, shelter_contact
    "homeostasis": [13, 14, 15, 16, 17],  # energy, damage, fatigue, novelty, stress
}
WORKSPACE_NUM_CHANNELS = len(WORKSPACE_CHANNELS)
WORKSPACE_CHANNEL_INDICES = list(WORKSPACE_CHANNELS.values())


class GlobalWorkspace(nn.Module):
    """Limited-capacity broadcast: channels compete for workspace access via attention."""

    def __init__(self, observation_size: int, hidden_size: int, num_channels: int = 4) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_encoders = nn.ModuleList()
        for indices in WORKSPACE_CHANNEL_INDICES:
            self.channel_encoders.append(
                nn.Sequential(nn.Linear(len(indices), hidden_size), nn.Tanh())
            )
        self.salience = nn.Linear(hidden_size, 1)
        self.broadcast_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, observation: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
            broadcast: (batch, hidden_size) — the winning channel's representation
            attention_weights: (batch, num_channels) — competition result
        """
        channel_reps = []
        for i, indices in enumerate(WORKSPACE_CHANNEL_INDICES):
            idx = torch.tensor(indices, device=observation.device)
            channel_obs = observation.index_select(-1, idx)
            channel_reps.append(self.channel_encoders[i](channel_obs))

        stacked = torch.stack(channel_reps, dim=1)  # (batch, num_channels, hidden)
        scores = self.salience(stacked + hidden.unsqueeze(1)).squeeze(-1)  # (batch, num_channels)
        weights = torch.softmax(scores, dim=-1)  # (batch, num_channels)
        broadcast = (weights.unsqueeze(-1) * stacked).sum(1)  # (batch, hidden)
        return self.broadcast_proj(broadcast), weights


class WorldModel(nn.Module):
    """Predicts next GRU hidden state from current hidden state and action."""

    def __init__(self, hidden_size: int, action_size: int, action_embed_size: int = 16) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(action_size, action_embed_size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size + action_embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden: Tensor, action: Tensor) -> Tensor:
        action_emb = self.action_embed(action)
        combined = torch.cat([hidden, action_emb], dim=-1)
        return self.predictor(combined)

    def prediction_error(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return (predicted - actual.detach()).pow(2).mean(dim=-1)


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        episodic_memory_slots: int = 0,
        use_global_workspace: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
        )
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.episodic_memory = (
            EpisodicMemory(hidden_size, episodic_memory_slots)
            if episodic_memory_slots > 0
            else None
        )
        self.workspace = (
            GlobalWorkspace(observation_size, hidden_size)
            if use_global_workspace
            else None
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.rnn.hidden_size, device=device)

    def forward_step(
        self,
        observation: Tensor,
        hidden: Tensor,
        memory_buffer: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        encoded = self.encoder(observation)
        next_hidden = self.rnn(encoded, hidden)
        output = next_hidden
        if self.episodic_memory is not None and memory_buffer is not None:
            output = self.episodic_memory(next_hidden, memory_buffer)
        workspace_weights = None
        if self.workspace is not None:
            broadcast, workspace_weights = self.workspace(observation, output)
            output = output + broadcast
        logits = self.policy_head(output)
        value = self.value_head(output).squeeze(-1)
        return next_hidden, logits, value, workspace_weights


class ReflexController:
    """Hard-coded survival overrides to keep online learning stable."""

    def override(self, observation: np.ndarray, proposed_action: int) -> tuple[int, bool]:
        action = Action(proposed_action)
        wall_left = observation[ObsIndex.WALL_LEFT]
        wall_center = observation[ObsIndex.WALL_CENTER]
        wall_right = observation[ObsIndex.WALL_RIGHT]
        hazard_left = observation[ObsIndex.HAZARD_LEFT]
        hazard_center = observation[ObsIndex.HAZARD_CENTER]
        hazard_right = observation[ObsIndex.HAZARD_RIGHT]
        food_contact = observation[ObsIndex.FOOD_CONTACT] > 0.5
        shelter_contact = observation[ObsIndex.SHELTER_CONTACT] > 0.5
        energy = observation[ObsIndex.ENERGY]
        fatigue = observation[ObsIndex.FATIGUE]

        if energy < 0.45 and food_contact and action != Action.EAT:
            return int(Action.EAT), True
        if fatigue > 0.82 and shelter_contact and action != Action.REST:
            return int(Action.REST), True
        if hazard_center > 0.62 and action in {Action.FORWARD, Action.EAT, Action.REST}:
            return self._safer_turn(hazard_left, hazard_right), True
        if wall_center > 0.7 and action == Action.FORWARD:
            return self._safer_turn(wall_left, wall_right), True
        return int(action), False

    @staticmethod
    def _safer_turn(left_intensity: float, right_intensity: float) -> int:
        return int(Action.TURN_RIGHT if left_intensity >= right_intensity else Action.TURN_LEFT)


@dataclass
class PolicyStep:
    hidden: Tensor
    value: Tensor
    log_prob: Tensor
    entropy: Tensor
    action: int
    raw_action: int
    reflex_override: bool


@dataclass
class EpisodeRecord:
    observations: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": np.asarray(self.observations, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
        }


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.episodes: deque[dict[str, np.ndarray]] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: EpisodeRecord) -> None:
        if not episode.actions:
            return
        self.episodes.append(episode.as_arrays())

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        burn_in: int,
    ) -> list[dict[str, np.ndarray | int]]:
        if not self.episodes:
            return []

        batch = []
        episodes = list(self.episodes)
        for _ in range(batch_size):
            episode = self.rng.choice(episodes)
            episode_length = int(len(episode["actions"]))
            if episode_length <= 0:
                continue
            train_len = min(seq_len, episode_length)
            start_max = max(episode_length - train_len, 0)
            start = self.rng.randint(0, start_max) if start_max > 0 else 0
            burn_start = max(0, start - burn_in)
            actual_burn = start - burn_start
            stop = start + train_len
            batch.append(
                {
                    "observations": episode["observations"][burn_start : stop + 1],
                    "actions": episode["actions"][start:stop],
                    "rewards": episode["rewards"][start:stop],
                    "dones": episode["dones"][start:stop],
                    "burn_in": actual_burn,
                }
            )
        return batch


class OrganismLearner:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        agent_config: AgentConfig,
        training_config: TrainingConfig,
        seed: int,
    ) -> None:
        self.device = torch.device("cpu")
        self.agent_config = agent_config
        self.training_config = training_config
        memory_slots = agent_config.episodic_memory_slots if agent_config.use_episodic_memory else 0
        self.model = RecurrentActorCritic(
            observation_size, action_size, agent_config.hidden_size,
            episodic_memory_slots=memory_slots,
            use_global_workspace=agent_config.use_global_workspace,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=agent_config.learning_rate)
        self.world_model = WorldModel(agent_config.hidden_size, action_size)
        self.world_model.to(self.device)
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=agent_config.wm_learning_rate
        )
        self.reflex = ReflexController()
        self.memory_buffer: MemoryBuffer | None = (
            MemoryBuffer(agent_config.episodic_memory_slots, agent_config.hidden_size)
            if agent_config.use_episodic_memory
            else None
        )
        self.replay = ReplayBuffer(training_config.replay_capacity, seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def initial_hidden(self, batch_size: int = 1) -> Tensor:
        return self.model.initial_hidden(batch_size, self.device)

    def reset_memory(self) -> None:
        if self.memory_buffer is not None:
            self.memory_buffer.clear()

    def _mem_tensor(self) -> Tensor | None:
        if self.memory_buffer is None:
            return None
        return self.memory_buffer.get_tensor(self.device)

    def select_action(
        self,
        observation: np.ndarray,
        hidden: Tensor,
        deterministic: bool = False,
        track_grad: bool = True,
    ) -> PolicyStep:
        self.model.train(not deterministic)
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mem = self._mem_tensor()
        use_no_grad = deterministic or not track_grad
        if use_no_grad:
            with torch.no_grad():
                next_hidden, logits, value, _ = self.model.forward_step(observation_tensor, hidden, memory_buffer=mem)
                dist = Categorical(logits=logits)
                raw_action_tensor = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                raw_action = int(raw_action_tensor.item())
                action, reflex_override = self.reflex.override(observation, raw_action)
                executed_action_tensor = torch.tensor([action], dtype=torch.int64, device=self.device)
                log_prob = dist.log_prob(executed_action_tensor)
                entropy = dist.entropy()
        else:
            next_hidden, logits, value, _ = self.model.forward_step(observation_tensor, hidden, memory_buffer=mem)
            dist = Categorical(logits=logits)
            raw_action_tensor = dist.sample()
            raw_action = int(raw_action_tensor.item())
            action, reflex_override = self.reflex.override(observation, raw_action)
            executed_action_tensor = torch.tensor([action], dtype=torch.int64, device=self.device)
            log_prob = dist.log_prob(executed_action_tensor)
            entropy = dist.entropy()
        if self.memory_buffer is not None:
            self.memory_buffer.push(next_hidden)
        return PolicyStep(
            hidden=next_hidden,
            value=value,
            log_prob=log_prob,
            entropy=entropy,
            action=action,
            raw_action=raw_action,
            reflex_override=reflex_override,
        )

    def advance_hidden(self, observation: np.ndarray, hidden: Tensor) -> Tensor:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mem = self._mem_tensor()
        with torch.no_grad():
            next_hidden, _, _, _ = self.model.forward_step(observation_tensor, hidden, memory_buffer=mem)
        if self.memory_buffer is not None:
            self.memory_buffer.push(next_hidden)
        return next_hidden.detach()

    def bootstrap_value(self, next_observation: np.ndarray, hidden: Tensor) -> Tensor:
        observation_tensor = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mem = self._mem_tensor()
        with torch.no_grad():
            _, _, next_value, _ = self.model.forward_step(observation_tensor, hidden, memory_buffer=mem)
        return next_value

    def compute_surprise(
        self, prev_hidden: Tensor, action: int, actual_hidden: Tensor
    ) -> float:
        with torch.no_grad():
            action_t = torch.tensor([action], dtype=torch.int64, device=self.device)
            predicted = self.world_model(prev_hidden.detach(), action_t)
            error = (predicted - actual_hidden.detach()).pow(2).mean()
        return float(error.item())

    def online_update(
        self,
        step: PolicyStep,
        reward: float,
        done: bool,
        next_value: Tensor,
        prev_hidden: Tensor | None = None,
    ) -> dict[str, float]:
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_mask = torch.tensor([0.0 if done else 1.0], dtype=torch.float32, device=self.device)
        target = reward_tensor + self.agent_config.gamma * done_mask * next_value
        advantage = target - step.value

        policy_loss = -(step.log_prob * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy_bonus = step.entropy.mean()
        loss = (
            policy_loss
            + self.agent_config.value_coef * value_loss
            - self.agent_config.entropy_coef * entropy_bonus
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.agent_config.grad_clip)
        self.optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }

        if prev_hidden is not None:
            action_t = torch.tensor([step.action], dtype=torch.int64, device=self.device)
            predicted = self.world_model(prev_hidden.detach(), action_t)
            wm_loss = (predicted - step.hidden.detach()).pow(2).mean()
            self.wm_optimizer.zero_grad(set_to_none=True)
            wm_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.agent_config.grad_clip)
            self.wm_optimizer.step()
            stats["wm_loss"] = float(wm_loss.item())

        return stats

    def sleep_update(self) -> dict[str, float]:
        if len(self.replay) < self.training_config.min_replay_episodes:
            return {}

        batches_run = 0
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_wm_loss = 0.0

        for _ in range(self.training_config.sleep_batches):
            batch = self.replay.sample_batch(
                batch_size=self.training_config.sleep_batch_size,
                seq_len=self.training_config.sleep_seq_len,
                burn_in=self.training_config.sleep_burn_in,
            )
            if not batch:
                continue

            segment_losses = []
            segment_policy_losses = []
            segment_value_losses = []
            segment_entropies = []
            segment_wm_losses = []
            for segment in batch:
                observations = torch.as_tensor(
                    segment["observations"], dtype=torch.float32, device=self.device
                )
                actions = torch.as_tensor(segment["actions"], dtype=torch.int64, device=self.device)
                rewards = torch.as_tensor(segment["rewards"], dtype=torch.float32, device=self.device)
                dones = torch.as_tensor(segment["dones"], dtype=torch.float32, device=self.device)
                burn_in = int(segment["burn_in"])
                hidden = self.initial_hidden()
                local_mem = (
                    MemoryBuffer(self.agent_config.episodic_memory_slots, self.agent_config.hidden_size)
                    if self.memory_buffer is not None
                    else None
                )

                if burn_in > 0:
                    with torch.no_grad():
                        for obs in observations[:burn_in]:
                            mem_t = local_mem.get_tensor(self.device) if local_mem else None
                            hidden, _, _, _ = self.model.forward_step(obs.unsqueeze(0), hidden, memory_buffer=mem_t)
                            if local_mem is not None:
                                local_mem.push(hidden)

                train_observations = observations[burn_in:]
                values = []
                logits = []
                hiddens = []
                current_hidden = hidden
                for index, obs in enumerate(train_observations):
                    hiddens.append(current_hidden)
                    mem_t = local_mem.get_tensor(self.device) if local_mem else None
                    current_hidden, step_logits, step_value, _ = self.model.forward_step(
                        obs.unsqueeze(0), current_hidden, memory_buffer=mem_t
                    )
                    if local_mem is not None:
                        local_mem.push(current_hidden)
                    values.append(step_value.squeeze(0))
                    if index < len(actions):
                        logits.append(step_logits.squeeze(0))

                if len(values) != len(actions) + 1:
                    continue

                current_values = torch.stack(values[:-1])
                next_values = torch.stack(values[1:]).detach()
                logits_tensor = torch.stack(logits)
                dist = Categorical(logits=logits_tensor)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                targets = rewards + self.agent_config.gamma * (1.0 - dones) * next_values
                advantages = targets - current_values

                policy_loss = -(log_probs * advantages.detach()).mean()
                value_loss = advantages.pow(2).mean()
                loss = (
                    policy_loss
                    + self.agent_config.value_coef * value_loss
                    - self.agent_config.entropy_coef * entropy
                )
                segment_losses.append(loss)
                segment_policy_losses.append(policy_loss.detach())
                segment_value_losses.append(value_loss.detach())
                segment_entropies.append(entropy.detach())

                wm_loss_items = []
                for t in range(len(actions)):
                    h_curr = hiddens[t].detach()
                    h_next = hiddens[t + 1].detach()
                    pred = self.world_model(h_curr, actions[t : t + 1])
                    wm_loss_items.append((pred - h_next).pow(2).mean())
                if wm_loss_items:
                    segment_wm_losses.append(torch.stack(wm_loss_items).mean())

            if not segment_losses:
                continue

            batch_loss = torch.stack(segment_losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.agent_config.grad_clip)
            self.optimizer.step()

            if segment_wm_losses:
                batch_wm_loss = torch.stack(segment_wm_losses).mean()
                self.wm_optimizer.zero_grad(set_to_none=True)
                batch_wm_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.world_model.parameters(), self.agent_config.grad_clip
                )
                self.wm_optimizer.step()
                total_wm_loss += float(batch_wm_loss.item())

            batches_run += 1
            total_loss += float(batch_loss.item())
            total_policy_loss += float(torch.stack(segment_policy_losses).mean().item())
            total_value_loss += float(torch.stack(segment_value_losses).mean().item())
            total_entropy += float(torch.stack(segment_entropies).mean().item())

        if batches_run == 0:
            return {}

        return {
            "loss": total_loss / batches_run,
            "policy_loss": total_policy_loss / batches_run,
            "value_loss": total_value_loss / batches_run,
            "entropy": total_entropy / batches_run,
            "wm_loss": total_wm_loss / batches_run,
            "batches": float(batches_run),
        }

    def save(self, path: str) -> None:
        torch.save(
            {"model": self.model.state_dict(), "world_model": self.world_model.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"], strict=False)
            if "world_model" in state:
                self.world_model.load_state_dict(state["world_model"])
        else:
            self.model.load_state_dict(state, strict=False)
