from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent import OrganismLearner
from .config import ExperimentConfig
from .env import OBSERVATION_SIZE, Action, ObsIndex, OrganismEnv


@dataclass
class StepSnapshot:
    action_name: str = "none"
    reward: float = 0.0
    total_return: float = 0.0
    reflex_override: bool = False
    done: bool = False
    death_reason: str | None = None
    info: dict[str, Any] | None = None


class SimulationSession:
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        checkpoint: Path | None = None,
        seed: int = 11,
        deterministic: bool = False,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.seed = seed
        self.deterministic = deterministic
        self.checkpoint: Path | None = None
        self.env = OrganismEnv(self.config.env, seed=seed)
        self.learner: OrganismLearner | None = None
        self.hidden = None
        self.observation: np.ndarray | None = None
        self.last_step = StepSnapshot()
        self.last_surprise: float = 0.0
        self.last_ownership: dict[str, float] = {}
        self.last_workspace_weights: list[float] = []
        self.last_confidence: float = 0.5
        self.last_narration: dict = {}
        self.food_eaten: int = 0
        self.last_ate_food: bool = False
        self.position_history: list[np.ndarray] = []

        if checkpoint is not None:
            self.load_checkpoint(checkpoint, reset=False)
        self.reset(seed=seed)

    def load_checkpoint(
        self,
        checkpoint: Path | None,
        *,
        config: ExperimentConfig | None = None,
        reset: bool = True,
    ) -> None:
        if checkpoint is None:
            if config is not None:
                self.config = config
                self.env = OrganismEnv(self.config.env, seed=self.seed)
            self.learner = None
            self.checkpoint = None
            if reset:
                self.reset(seed=self.seed)
            return

        resolved = checkpoint.resolve()
        loaded_config = config or load_config(None, resolved)
        self.config = loaded_config
        self.env = OrganismEnv(self.config.env, seed=self.seed)
        self.learner = OrganismLearner(
            observation_size=OBSERVATION_SIZE,
            action_size=len(Action),
            agent_config=self.config.agent,
            training_config=self.config.train,
            seed=self.config.train.seed,
        )
        self.learner.load(str(resolved))
        self.checkpoint = resolved
        if reset:
            self.reset(seed=self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.observation = self.env.reset(seed=self.seed)
        self.hidden = self.learner.initial_hidden() if self.learner else None
        if self.learner is not None:
            self.learner.reset_memory()
        self.last_step = StepSnapshot()
        self.food_eaten = 0
        self.last_ate_food = False
        self.position_history = [self.env.agent_position.copy()]

    def randomize_seed(self) -> int:
        seed = random.randint(0, 1_000_000)
        self.reset(seed=seed)
        return seed

    def set_deterministic(self, deterministic: bool) -> None:
        self.deterministic = deterministic

    def step_policy(self) -> StepSnapshot:
        if self.learner is None:
            raise RuntimeError("No checkpoint loaded; policy stepping is unavailable.")
        if self.observation is None:
            self.reset(seed=self.seed)
        prev_hidden = self.hidden
        prev_obs = self.observation
        policy_step = self.learner.select_action(
            self.observation,
            self.hidden,
            deterministic=self.deterministic,
            track_grad=False,
        )
        self.hidden = policy_step.hidden.detach()
        self.last_surprise = self.learner.compute_surprise(
            prev_hidden, policy_step.action, self.hidden
        )
        snapshot = self._apply_action(policy_step.action, policy_step.reflex_override)
        self.last_ownership = self.learner.compute_ownership(
            prev_obs, policy_step.action, prev_hidden, self.observation
        )
        intro = self.learner.introspect(self.observation, self.hidden)
        self.last_workspace_weights = intro.get("workspace_weights", [])
        self.last_confidence = intro.get("confidence", 0.5)
        self.last_narration = self.learner.narrate(self.observation, self.hidden)
        return snapshot

    def step_manual(self, action: Action) -> StepSnapshot:
        if self.observation is None:
            self.reset(seed=self.seed)
        if self.learner is not None:
            self.hidden = self.learner.advance_hidden(self.observation, self.hidden)
        return self._apply_action(int(action), reflex_override=False)

    def state_dict(self) -> dict[str, Any]:
        if self.observation is None:
            self.reset(seed=self.seed)

        env = self.env
        observation = self.observation
        available_food = [
            _point_to_list(point)
            for point, cooldown in zip(env.food_positions, env.food_cooldowns)
            if cooldown == 0 and np.all(point >= 0.0)
        ]

        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "has_policy": self.learner is not None,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_name": self.checkpoint.parent.name if self.checkpoint else None,
            "episode": {
                "steps": env.episode_steps,
                "return": float(env.episode_return),
                "energy": float(env.energy),
                "damage": float(env.damage),
                "fatigue": float(env.fatigue),
                "stress": float(env._discomfort()),
                "done": self.last_step.done,
                "death_reason": self.last_step.death_reason,
                "action_name": self.last_step.action_name,
                "reward": float(self.last_step.reward),
                "reflex_override": self.last_step.reflex_override,
                "surprise": self.last_surprise,
                "confidence": self.last_confidence,
                "food_eaten": self.food_eaten,
                "ate_food": self.last_ate_food,
            },
            "world": {
                "size": float(env.config.world_size),
                "sensor_range": float(env.config.sensor_range),
                "eat_radius": float(env.config.eat_radius),
                "shelter_radius": float(env.config.shelter_radius),
                "hazard_radius": float(env.config.hazard_radius),
                "agent": _point_to_list(env.agent_position),
                "heading": float(env.heading),
                "shelter": _point_to_list(env.shelter_position),
                "food": available_food,
                "hazards": [_point_to_list(point) for point in env.hazard_positions],
                "trail": [_point_to_list(point) for point in self.position_history[-160:]],
                "visitation": env.visitation.astype(int).tolist(),
                "sector_offsets": [float(offset) for offset in env.sector_offsets],
                "food_visible_range": float(env.config.food_visible_range),
            },
            "cognition": {
                "workspace_weights": self.last_workspace_weights,
                "workspace_channels": ["food", "danger", "shelter", "homeostasis"],
                "ownership": self.last_ownership,
                "memory_slots_used": len(self.learner.memory_buffer.buffer) if self.learner and self.learner.memory_buffer else 0,
                "memory_slots_total": self.config.agent.episodic_memory_slots if self.config.agent.use_episodic_memory else 0,
                "narration": self.last_narration,
            },
            "sensors": {
                "food": _triplet(
                    observation[ObsIndex.FOOD_LEFT],
                    observation[ObsIndex.FOOD_CENTER],
                    observation[ObsIndex.FOOD_RIGHT],
                ),
                "hazard": _triplet(
                    observation[ObsIndex.HAZARD_LEFT],
                    observation[ObsIndex.HAZARD_CENTER],
                    observation[ObsIndex.HAZARD_RIGHT],
                ),
                "wall": _triplet(
                    observation[ObsIndex.WALL_LEFT],
                    observation[ObsIndex.WALL_CENTER],
                    observation[ObsIndex.WALL_RIGHT],
                ),
                "shelter": {
                    "alignment": float(observation[ObsIndex.SHELTER_ALIGNMENT]),
                    "proximity": float(observation[ObsIndex.SHELTER_PROXIMITY]),
                    "contact": float(observation[ObsIndex.SHELTER_CONTACT]),
                },
                "internal": {
                    "food_contact": float(observation[ObsIndex.FOOD_CONTACT]),
                    "energy": float(observation[ObsIndex.ENERGY]),
                    "damage": float(observation[ObsIndex.DAMAGE]),
                    "fatigue": float(observation[ObsIndex.FATIGUE]),
                    "novelty": float(observation[ObsIndex.NOVELTY_DRIVE]),
                    "stress": float(observation[ObsIndex.STRESS]),
                },
            },
        }

    def _apply_action(self, action: int, reflex_override: bool) -> StepSnapshot:
        self.observation, reward, done, info = self.env.step(action)
        self.last_ate_food = info.get("ate_food", False)
        if self.last_ate_food:
            self.food_eaten += 1
        self.position_history.append(self.env.agent_position.copy())
        if len(self.position_history) > 160:
            self.position_history = self.position_history[-160:]
        snapshot = StepSnapshot(
            action_name=info["action_name"],
            reward=reward,
            total_return=self.env.episode_return,
            reflex_override=reflex_override,
            done=done,
            death_reason=info["death_reason"],
            info=info,
        )
        self.last_step = snapshot
        return snapshot


def load_config(config_path: Path | None, checkpoint_path: Path | None) -> ExperimentConfig:
    candidates = []
    if config_path is not None:
        candidates.append(config_path)
    if checkpoint_path is not None:
        candidates.append(checkpoint_path.with_name("config.json"))

    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if "config" in payload:
                payload = payload["config"]
            return ExperimentConfig.from_dict(payload)
    return ExperimentConfig()


def discover_checkpoints(outputs_dir: Path) -> list[dict[str, str]]:
    if not outputs_dir.exists():
        return []

    discovered = []
    for checkpoint in outputs_dir.glob("*/model.pt"):
        try:
            stat = checkpoint.stat()
        except OSError:
            continue
        discovered.append(
            {
                "name": checkpoint.parent.name,
                "path": str(checkpoint.resolve()),
                "config": str(checkpoint.with_name("config.json").resolve()),
                "modified": str(int(stat.st_mtime)),
            }
        )

    discovered.sort(key=lambda item: (item["name"] != "viewer_baseline", -int(item["modified"])))
    return discovered


def _triplet(left: float, center: float, right: float) -> dict[str, float]:
    return {
        "left": float(left),
        "center": float(center),
        "right": float(right),
    }


def _point_to_list(point: np.ndarray) -> list[float]:
    return [float(point[0]), float(point[1])]
