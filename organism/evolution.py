from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .agent import RecurrentActorCritic
from .config import EnvironmentConfig, EvolutionConfig
from .env import OBSERVATION_SIZE, Action, ObsIndex
from .evo_genetics import (
    crossover_physical,
    crossover_visual,
    crossover_weights,
    mutate_physical,
    mutate_visual,
    mutate_weights,
    seed_population,
)


@dataclass
class VisualTraits:
    color_h: float      # hue [0, 360]
    color_s: float      # saturation [0.3, 1.0]
    color_l: float      # lightness [0.3, 0.8]
    pattern: str        # "solid"|"stripe"|"spot"|"ring"
    body_size: float    # [0.6, 1.8]
    shape: str          # "circle"|"triangle"|"diamond"


@dataclass
class PhysicalTraits:
    sensor_range: float
    food_visible_range: float
    move_speed: float
    turn_angle: float
    fov_half_angle: float       # half-width of sensor cone in radians
    energy_decay: float
    eat_radius: float
    hazard_sensitivity: float   # scales hazard detection radius


@dataclass
class OrganismState:
    uid: int
    lineage: int
    generation: int
    position: np.ndarray        # (2,) float32
    heading: float
    energy: float
    damage: float
    fatigue: float
    age: int
    food_eaten: int
    physical: PhysicalTraits
    visual: VisualTraits
    model_sd: dict              # RecurrentActorCritic weights (CPU tensors)
    hidden: Tensor              # (1, hidden_size) on CPU
    observation: np.ndarray
    visitation: np.ndarray      # 8×8 int32 per-organism novelty grid
    last_visit_cell: tuple | None
    last_action: str

    def is_mate_ready(self, cfg: "EvolutionConfig") -> bool:
        return (
            self.age >= cfg.min_mate_age
            and self.food_eaten >= cfg.mate_food_min
            and self.energy >= cfg.mate_energy_min
            and self.damage <= cfg.mate_damage_max
            and self.fatigue <= cfg.mate_fatigue_max
        )

    def to_json(self, cfg: "EvolutionConfig | None" = None) -> dict[str, Any]:
        p = self.physical
        v = self.visual
        mate_ready = self.is_mate_ready(cfg) if cfg is not None else False
        return {
            "uid": self.uid,
            "lineage": self.lineage,
            "generation": self.generation,
            "position": [float(self.position[0]), float(self.position[1])],
            "heading": float(self.heading),
            "energy": float(self.energy),
            "damage": float(self.damage),
            "fatigue": float(self.fatigue),
            "age": self.age,
            "food_eaten": self.food_eaten,
            "last_action": self.last_action,
            "mate_ready": mate_ready,
            "physical": {
                "sensor_range": p.sensor_range,
                "food_visible_range": p.food_visible_range,
                "move_speed": p.move_speed,
                "turn_angle": p.turn_angle,
                "fov_half_angle": p.fov_half_angle,
                "energy_decay": p.energy_decay,
                "eat_radius": p.eat_radius,
                "hazard_sensitivity": p.hazard_sensitivity,
            },
            "visual": {
                "color_h": v.color_h,
                "color_s": v.color_s,
                "color_l": v.color_l,
                "pattern": v.pattern,
                "body_size": v.body_size,
                "shape": v.shape,
            },
        }


@dataclass
class EggState:
    uid: int
    position: np.ndarray
    lineage_a: int
    lineage_b: int
    generation: int
    hatch_countdown: int
    physical: PhysicalTraits
    visual: VisualTraits
    model_sd: dict

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "position": [float(self.position[0]), float(self.position[1])],
            "lineage_a": self.lineage_a,
            "lineage_b": self.lineage_b,
            "generation": self.generation,
            "hatch_countdown": self.hatch_countdown,
        }


_NOVELTY_GRID = 8
_HIDDEN_SIZE = 256
_SECTOR_OFFSETS = np.array([-0.9, 0.0, 0.9], dtype=np.float32)


def _wrap_angle(a: float) -> float:
    return float(((a + math.pi) % (2 * math.pi)) - math.pi)


def _angle_to(vector: np.ndarray) -> float:
    return float(np.arctan2(vector[1], vector[0]))


def _sector_response(
    agent_pos: np.ndarray,
    heading: float,
    sector_offsets: np.ndarray,
    targets: np.ndarray,
    effective_range: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    response = np.zeros(3, dtype=np.float32)
    if len(targets) == 0:
        return response
    for i, target in enumerate(targets):
        vector = target - agent_pos
        dist = float(np.linalg.norm(vector))
        if dist <= 1e-6 or dist > effective_range:
            continue
        w = float(weights[i]) if weights is not None else 1.0
        rel_angle = _wrap_angle(_angle_to(vector) - heading)
        dist_weight = 1.0 - dist / effective_range
        for idx, center in enumerate(sector_offsets):
            angular = max(0.0, np.cos(rel_angle - center))
            response[idx] += w * dist_weight * angular
    return np.clip(response, 0.0, 1.0)


def _wall_sensor(agent_pos: np.ndarray, heading: float, angle: float, world_size: float, sensor_range: float) -> float:
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    distances = []
    for axis in range(2):
        component = direction[axis]
        if abs(component) < 1e-6:
            continue
        if component > 0.0:
            distances.append((world_size - agent_pos[axis]) / component)
        else:
            distances.append(-agent_pos[axis] / component)
    ray_distance = min(d for d in distances if d >= 0.0) if distances else sensor_range
    return float(np.clip(1.0 - min(ray_distance / sensor_range, 1.0), 0.0, 1.0))


def _mark_visited(org: OrganismState, world_size: float = 1.0) -> float:
    scaled = np.clip(org.position / world_size, 0.0, 0.9999)
    x = int(scaled[0] * _NOVELTY_GRID)
    y = int(scaled[1] * _NOVELTY_GRID)
    visits = org.visitation[y, x]
    org.visitation[y, x] += 1
    current_cell = (x, y)
    moved = current_cell != org.last_visit_cell
    org.last_visit_cell = current_cell
    if not moved:
        return 0.0
    return 0.0  # no novelty reward in evolution mode (pure survival pressure)


def _compute_observation(
    org: OrganismState,
    food_positions: np.ndarray,
    food_cooldowns: np.ndarray,
    food_values: np.ndarray,
    hazard_positions: np.ndarray,
    hazard_values: np.ndarray,
    shelter_position: np.ndarray,
    world_size: float = 1.0,
) -> np.ndarray:
    p = org.physical
    sector_offsets = np.array([-p.fov_half_angle, 0.0, p.fov_half_angle], dtype=np.float32)

    available_mask = food_cooldowns == 0
    available_food = food_positions[available_mask]
    available_food_values = food_values[available_mask]

    food_sensors = _sector_response(
        org.position, org.heading, sector_offsets, available_food,
        p.food_visible_range, available_food_values,
    )
    hazard_sensors = _sector_response(
        org.position, org.heading, sector_offsets, hazard_positions,
        p.sensor_range * p.hazard_sensitivity, hazard_values,
    )
    wall_sensors = np.array(
        [_wall_sensor(org.position, org.heading, org.heading + o, world_size, p.sensor_range)
         for o in sector_offsets],
        dtype=np.float32,
    )

    shelter_vec = shelter_position - org.position
    shelter_dist = float(np.linalg.norm(shelter_vec))
    shelter_align = float(0.5 * (np.cos(_angle_to(shelter_vec) - org.heading) + 1.0))
    shelter_prox = float(1.0 - min(shelter_dist / p.sensor_range, 1.0))

    # Food contact: nearest food within eat_radius
    food_contact = 0.0
    if len(available_food) > 0:
        dists = np.linalg.norm(available_food - org.position, axis=1)
        if dists.min() <= p.eat_radius:
            food_contact = 1.0
    shelter_contact = 1.0 if shelter_dist <= 0.12 else 0.0

    x, y_coord = int(np.clip(org.position[0] / world_size, 0, 0.9999) * _NOVELTY_GRID), \
                 int(np.clip(org.position[1] / world_size, 0, 0.9999) * _NOVELTY_GRID)
    novelty_drive = float(1.0 / (1.0 + org.visitation[y_coord, x]))

    discomfort = float(np.clip(
        0.55 * (1.0 - org.energy) + 0.35 * org.damage + 0.25 * org.fatigue, 0.0, 1.5
    ))
    stress = float(np.clip(discomfort, 0.0, 1.0))

    obs = np.concatenate([
        food_sensors,
        hazard_sensors,
        wall_sensors,
        np.array([
            shelter_align, shelter_prox, food_contact, shelter_contact,
            org.energy, org.damage, org.fatigue, novelty_drive, stress,
        ], dtype=np.float32),
    ])
    return obs.astype(np.float32)


class EvolutionSimulation:
    def __init__(
        self,
        config: EvolutionConfig | None = None,
        env_config: EnvironmentConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or EvolutionConfig()
        self.env_config = env_config or EnvironmentConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._uid_counter = 0

        # Shared world
        self.food_positions = np.zeros((self.env_config.num_food_sources, 2), dtype=np.float32)
        self.food_cooldowns = np.zeros(self.env_config.num_food_sources, dtype=np.int32)
        self.food_values = np.ones(self.env_config.num_food_sources, dtype=np.float32)
        self.hazard_positions = np.zeros((self.env_config.num_hazards, 2), dtype=np.float32)
        self.hazard_values = np.ones(self.env_config.num_hazards, dtype=np.float32)
        self.shelter_position = np.array([0.5, 0.5], dtype=np.float32)

        # Population
        self.organisms: list[OrganismState] = []
        self.eggs: list[EggState] = []

        # Stats
        self.tick = 0
        self.total_born = 0
        self.total_died = 0
        self.mating_events = 0
        self.trait_history: list[dict[str, float]] = []

        self._reset_world()
        self._spawn_initial_population()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._step_once()

    def state_dict(self) -> dict[str, Any]:
        alive = self.organisms
        eggs = self.eggs
        available = [
            ([float(p[0]), float(p[1])], float(v))
            for p, c, v in zip(self.food_positions, self.food_cooldowns, self.food_values)
            if c == 0 and np.all(p >= 0.0)
        ]
        food_pos = [a[0] for a in available]
        food_vals = [a[1] for a in available]

        dominant = self._dominant_lineage()
        max_gen = max((o.generation for o in alive), default=0)

        trait_stats: dict[str, float] = {}
        if alive:
            speeds = [o.physical.move_speed for o in alive]
            sights = [o.physical.food_visible_range for o in alive]
            sizes = [o.visual.body_size for o in alive]
            trait_stats = {
                "mean_speed": float(np.mean(speeds)),
                "std_speed": float(np.std(speeds)),
                "mean_sight": float(np.mean(sights)),
                "std_sight": float(np.std(sights)),
                "mean_size": float(np.mean(sizes)),
                "std_size": float(np.std(sizes)),
                "mean_generation": float(np.mean([o.generation for o in alive])),
            }

        return {
            "tick": self.tick,
            "alive_count": len(alive),
            "egg_count": len(eggs),
            "total_born": self.total_born,
            "total_died": self.total_died,
            "mating_events": self.mating_events,
            "generation_max": max_gen,
            "dominant_lineage": dominant,
            "world": {
                "size": float(self.env_config.world_size),
                "shelter": [float(self.shelter_position[0]), float(self.shelter_position[1])],
                "shelter_radius": float(self.env_config.shelter_radius),
                "food": food_pos,
                "food_values": food_vals,
                "hazards": [[float(p[0]), float(p[1])] for p in self.hazard_positions],
                "hazard_values": [float(v) for v in self.hazard_values],
                "hazard_radius": float(self.env_config.hazard_radius),
                "eat_radius": float(self.env_config.eat_radius),
            },
            "organisms": [o.to_json(self.config) for o in alive],
            "eggs": [e.to_json() for e in eggs],
            "trait_stats": trait_stats,
            "mate_conditions": {
                "min_age": self.config.min_mate_age,
                "min_food": self.config.mate_food_min,
                "min_energy": self.config.mate_energy_min,
            },
        }

    def reset(self, seed: int | None = None, population_size: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        if population_size is not None:
            self.config.population_size = population_size
        self.rng = np.random.default_rng(self.seed)
        self._uid_counter = 0
        self.organisms = []
        self.eggs = []
        self.tick = 0
        self.total_born = 0
        self.total_died = 0
        self.mating_events = 0
        self.trait_history = []
        self._reset_world()
        self._spawn_initial_population()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_uid(self) -> int:
        uid = self._uid_counter
        self._uid_counter += 1
        return uid

    def _reset_world(self) -> None:
        ws = self.env_config.world_size
        for i in range(self.env_config.num_food_sources):
            self.food_positions[i] = self._spawn_point(0.15, ws)
            self.food_values[i] = self._sample_value()
        for i in range(self.env_config.num_hazards):
            self.hazard_positions[i] = self._spawn_point(0.18, ws)
            self.hazard_values[i] = self._sample_value()
        self.food_cooldowns.fill(0)

    def _spawn_initial_population(self) -> None:
        cfg = self.config
        checkpoint = cfg.seed_checkpoint or self._find_checkpoint()
        genomes = seed_population(checkpoint, cfg.population_size, cfg.initial_diversity_sigma, self.rng)
        for i, (sd, phys, vis) in enumerate(genomes):
            self.organisms.append(self._make_organism(
                uid=self._next_uid(),
                lineage=i,
                generation=0,
                sd=sd,
                physical=phys,
                visual=vis,
            ))
            self.total_born += 1

    def _make_organism(
        self,
        uid: int,
        lineage: int,
        generation: int,
        sd: dict,
        physical: PhysicalTraits,
        visual: VisualTraits,
        position: np.ndarray | None = None,
    ) -> OrganismState:
        ws = self.env_config.world_size
        pos = position if position is not None else self._spawn_near_shelter(ws)
        return OrganismState(
            uid=uid,
            lineage=lineage,
            generation=generation,
            position=pos,
            heading=float(self.rng.uniform(-math.pi, math.pi)),
            energy=self.env_config.start_energy,
            damage=self.env_config.start_damage,
            fatigue=self.env_config.start_fatigue,
            age=0,
            food_eaten=0,
            physical=physical,
            visual=visual,
            model_sd=sd,
            hidden=torch.zeros(1, _HIDDEN_SIZE),
            observation=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            visitation=np.zeros((_NOVELTY_GRID, _NOVELTY_GRID), dtype=np.int32),
            last_visit_cell=None,
            last_action="none",
        )

    def _step_once(self) -> None:
        self.tick += 1
        self._tick_food_respawns()

        # Randomise step order to prevent positional food-eating bias
        order = list(range(len(self.organisms)))
        self.rng.shuffle(order)

        # Build per-organism model instances lazily (cache by uid to avoid rebuilding each tick)
        for idx in order:
            if idx >= len(self.organisms):
                continue
            org = self.organisms[idx]
            obs = _compute_observation(
                org,
                self.food_positions, self.food_cooldowns, self.food_values,
                self.hazard_positions, self.hazard_values,
                self.shelter_position, self.env_config.world_size,
            )
            org.observation = obs
            # Record distance to nearest ready mate before moving (for shaping)
            prev_mate_dist = self._nearest_ready_mate_dist(org) if self.config.mate_approach_scale > 0.0 else None
            action = self._policy_step(org, obs)
            org.last_action = list(Action.__members__.keys())[action].lower()
            self._apply_action(org, action)
            # Mate-approach shaping: reward getting closer to a potential mate when ready
            if prev_mate_dist is not None and org.is_mate_ready(self.config):
                curr_mate_dist = self._nearest_ready_mate_dist(org)
                if prev_mate_dist > 0.0 and curr_mate_dist > 0.0:
                    delta = prev_mate_dist - curr_mate_dist
                    # Small positive energy bonus (non-damaging encouragement)
                    org.energy = float(np.clip(
                        org.energy + self.config.mate_approach_scale * 0.01 * delta,
                        0.0, 1.0
                    ))

        # Hazard damage
        for org in self.organisms:
            self._apply_hazard_damage(org)

        # Death checks
        dead_indices = []
        for i, org in enumerate(self.organisms):
            if org.energy <= 0.0 or org.damage >= 1.0:
                dead_indices.append(i)
        for i in reversed(dead_indices):
            self.organisms.pop(i)
            self.total_died += 1

        # Egg hatching
        hatched = []
        remaining = []
        for egg in self.eggs:
            egg.hatch_countdown -= 1
            if egg.hatch_countdown <= 0:
                hatched.append(egg)
            else:
                remaining.append(egg)
        self.eggs = remaining
        for egg in hatched:
            self._hatch_egg(egg)

        # Mating
        if len(self.organisms) + len(self.eggs) < self.config.max_population:
            self._check_mating()

        # Population floor
        if len(self.organisms) < self.config.min_population:
            self._emergency_spawn()

        # Age all organisms
        for org in self.organisms:
            org.age += 1

        # Snapshot traits every 100 ticks
        if self.tick % 100 == 0 and self.organisms:
            self.trait_history.append({
                "tick": self.tick,
                "mean_speed": float(np.mean([o.physical.move_speed for o in self.organisms])),
                "mean_sight": float(np.mean([o.physical.food_visible_range for o in self.organisms])),
                "mean_size": float(np.mean([o.visual.body_size for o in self.organisms])),
            })
            if len(self.trait_history) > 500:
                self.trait_history = self.trait_history[-500:]

    def _policy_step(self, org: OrganismState, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        model = self._get_model(org)
        with torch.no_grad():
            hidden, logits, _, _ = model.forward_step(obs_t, org.hidden)
        org.hidden = hidden.detach()
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def _get_model(self, org: OrganismState) -> RecurrentActorCritic:
        """Build a RecurrentActorCritic from the organism's stored state_dict."""
        model = RecurrentActorCritic(
            observation_size=OBSERVATION_SIZE,
            action_size=len(Action),
            hidden_size=_HIDDEN_SIZE,
            use_global_workspace=True,
        )
        model.load_state_dict(org.model_sd, strict=False)
        model.eval()
        return model

    def _apply_action(self, org: OrganismState, action: int) -> None:
        p = org.physical
        ws = self.env_config.world_size
        chosen = Action(action)

        if chosen == Action.TURN_LEFT:
            org.heading -= p.turn_angle
            org.energy -= self.env_config.turn_energy_cost
            org.fatigue += 0.004
        elif chosen == Action.TURN_RIGHT:
            org.heading += p.turn_angle
            org.energy -= self.env_config.turn_energy_cost
            org.fatigue += 0.004
        elif chosen == Action.FORWARD:
            delta = np.array([np.cos(org.heading), np.sin(org.heading)], dtype=np.float32)
            next_pos = org.position + delta * p.move_speed
            clipped = np.clip(next_pos, 0.0, ws)
            collision = not np.allclose(clipped, next_pos)
            org.position = clipped
            org.energy -= self.env_config.movement_energy_cost
            org.fatigue += 0.018
            if collision:
                org.damage += 0.02
        elif chosen == Action.EAT:
            org.energy -= 0.002
            idx = self._nearest_food_within(org.position, p.eat_radius)
            if idx is not None:
                food_val = float(self.food_values[idx])
                org.energy += self.env_config.food_energy_gain * food_val
                org.food_eaten += 1
                self.food_cooldowns[idx] = self.env_config.food_respawn_steps
                self.food_positions[idx] = np.array([-1.0, -1.0], dtype=np.float32)
            org.fatigue += 0.004
        elif chosen == Action.REST:
            shelter_dist = float(np.linalg.norm(org.position - self.shelter_position))
            scale = 1.0 if shelter_dist <= self.env_config.shelter_radius else 0.5
            org.fatigue -= self.env_config.rest_recovery * scale
            org.damage -= self.env_config.passive_healing * scale
            org.energy += 0.006 * scale

        org.heading = _wrap_angle(org.heading)
        org.energy -= p.energy_decay
        org.fatigue += 0.004

        org.energy = float(np.clip(org.energy, 0.0, 1.0))
        org.damage = float(np.clip(org.damage, 0.0, 1.0))
        org.fatigue = float(np.clip(org.fatigue, 0.0, 1.0))
        _mark_visited(org, ws)

    def _apply_hazard_damage(self, org: OrganismState) -> None:
        for i, hazard in enumerate(self.hazard_positions):
            dist = float(np.linalg.norm(org.position - hazard))
            if dist <= self.env_config.hazard_radius:
                severity = 1.0 - (dist / self.env_config.hazard_radius)
                strength = float(self.hazard_values[i])
                org.damage += self.env_config.hazard_damage * (0.45 + 0.55 * severity) * strength
                org.energy -= 0.01 * severity * strength
        org.energy = float(np.clip(org.energy, 0.0, 1.0))
        org.damage = float(np.clip(org.damage, 0.0, 1.0))

    def _check_mating(self) -> None:
        cfg = self.config
        eligible = [
            o for o in self.organisms
            if o.age >= cfg.min_mate_age
            and o.energy >= cfg.mate_energy_min
            and o.damage <= cfg.mate_damage_max
            and o.fatigue <= cfg.mate_fatigue_max
            and o.food_eaten >= cfg.mate_food_min
        ]
        paired: set[int] = set()
        to_remove: list[int] = []

        for i, a in enumerate(eligible):
            if a.uid in paired:
                continue
            for b in eligible[i + 1:]:
                if b.uid in paired:
                    continue
                dist = float(np.linalg.norm(a.position - b.position))
                if dist <= cfg.mate_radius:
                    # Mate!
                    child_sd = mutate_weights(
                        crossover_weights(a.model_sd, b.model_sd, self.rng),
                        cfg.mutation_sigma, self.rng,
                    )
                    child_phys = mutate_physical(
                        crossover_physical(a.physical, b.physical, self.rng),
                        self.rng, scale=cfg.trait_mutation_scale,
                    )
                    child_vis = mutate_visual(
                        crossover_visual(a.visual, b.visual, self.rng),
                        self.rng,
                    )
                    midpoint = ((a.position + b.position) * 0.5).astype(np.float32)
                    self.eggs.append(EggState(
                        uid=self._next_uid(),
                        position=midpoint,
                        lineage_a=a.lineage,
                        lineage_b=b.lineage,
                        generation=max(a.generation, b.generation) + 1,
                        hatch_countdown=cfg.hatch_countdown,
                        physical=child_phys,
                        visual=child_vis,
                        model_sd=child_sd,
                    ))
                    paired.add(a.uid)
                    paired.add(b.uid)
                    self.mating_events += 1
                    # Auto-save both parents on successful mating
                    try:
                        self.save_organism(a, note=f"mated@tick{self.tick}")
                        self.save_organism(b, note=f"mated@tick{self.tick}")
                    except Exception:
                        pass
                    break

        # Remove mated organisms
        self.organisms = [o for o in self.organisms if o.uid not in paired]
        self.total_died += len(paired)  # they "die" to produce offspring

    def _hatch_egg(self, egg: EggState) -> None:
        ws = self.env_config.world_size
        angle = float(self.rng.uniform(-math.pi, math.pi))
        offset = 0.06 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        spawn_pos = np.clip(egg.position + offset, 0.05, ws - 0.05)
        # Lineage is the dominant parent lineage
        lineage = egg.lineage_a if self.rng.integers(0, 2) == 0 else egg.lineage_b
        org = self._make_organism(
            uid=self._next_uid(),
            lineage=lineage,
            generation=egg.generation,
            sd=egg.model_sd,
            physical=egg.physical,
            visual=egg.visual,
            position=spawn_pos,
        )
        self.organisms.append(org)
        self.total_born += 1

    def _emergency_spawn(self) -> None:
        """Spawn a mutated copy of the fittest surviving organism to prevent extinction."""
        if not self.organisms:
            self._spawn_initial_population()
            return
        best = max(self.organisms, key=lambda o: o.food_eaten * 10 + o.age)
        child_sd = mutate_weights(best.model_sd, self.config.mutation_sigma * 2, self.rng)
        child_phys = mutate_physical(best.physical, self.rng, scale=1.5)
        child_vis = mutate_visual(best.visual, self.rng)
        org = self._make_organism(
            uid=self._next_uid(),
            lineage=best.lineage,
            generation=best.generation + 1,
            sd=child_sd,
            physical=child_phys,
            visual=child_vis,
        )
        self.organisms.append(org)
        self.total_born += 1

    def _tick_food_respawns(self) -> None:
        ws = self.env_config.world_size
        for i in range(len(self.food_cooldowns)):
            if self.food_cooldowns[i] <= 0:
                continue
            self.food_cooldowns[i] -= 1
            if self.food_cooldowns[i] == 0:
                self.food_positions[i] = self._spawn_point(0.15, ws)
                self.food_values[i] = self._sample_value()

    def _nearest_ready_mate_dist(self, org: OrganismState) -> float:
        """Return distance to nearest other ready-to-mate organism, or 0 if none."""
        best = 0.0
        for other in self.organisms:
            if other.uid == org.uid:
                continue
            if not other.is_mate_ready(self.config):
                continue
            d = float(np.linalg.norm(org.position - other.position))
            if best == 0.0 or d < best:
                best = d
        return best

    def _nearest_food_within(self, position: np.ndarray, radius: float) -> int | None:
        available = np.where(self.food_cooldowns == 0)[0]
        if len(available) == 0:
            return None
        avail_pos = self.food_positions[available]
        dists = np.linalg.norm(avail_pos - position, axis=1)
        nearest = int(np.argmin(dists))
        if float(dists[nearest]) <= radius:
            return int(available[nearest])
        return None

    def _dominant_lineage(self) -> int:
        if not self.organisms:
            return 0
        from collections import Counter
        counts = Counter(o.lineage for o in self.organisms)
        return counts.most_common(1)[0][0]

    def _spawn_point(self, min_dist: float, ws: float) -> np.ndarray:
        for _ in range(256):
            point = self.rng.uniform(0.05, ws - 0.05, size=2).astype(np.float32)
            if float(np.linalg.norm(point - self.shelter_position)) >= min_dist:
                return point
        return self.rng.uniform(0.05, ws - 0.05, size=2).astype(np.float32)

    def _spawn_near_shelter(self, ws: float) -> np.ndarray:
        angle = float(self.rng.uniform(-math.pi, math.pi))
        offset = 0.08 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return np.clip(self.shelter_position + offset, 0.05, ws - 0.05)

    def _sample_value(self) -> float:
        if not self.env_config.variety:
            return 1.0
        tiers = [0.5, 1.0, 1.6, 2.3]
        weights = [0.3, 0.4, 0.2, 0.1]
        return float(self.rng.choice(tiers, p=weights))

    def _find_checkpoint(self) -> str | None:
        candidates = [
            Path("outputs/organism-v18b/model_best.pt"),
            Path("outputs/organism-v18/model_best.pt"),
            Path("outputs/organism-v17/model_best.pt"),
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        outputs = Path("outputs")
        if outputs.exists():
            best_pts = sorted(outputs.glob("*/model_best.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if best_pts:
                return str(best_pts[0])
        return None

    # ------------------------------------------------------------------
    # Save / Load / Spawn
    # ------------------------------------------------------------------

    @staticmethod
    def _save_dir() -> Path:
        d = Path("outputs/evolution/saved")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _index_path() -> Path:
        return Path("outputs/evolution/saved/index.json")

    @staticmethod
    def _load_index() -> list[dict[str, Any]]:
        p = EvolutionSimulation._index_path()
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    @staticmethod
    def _write_index(entries: list[dict[str, Any]]) -> None:
        p = EvolutionSimulation._index_path()
        p.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def save_organism(self, org: "OrganismState", note: str = "") -> str:
        """Persist an organism's genome to disk. Returns the save key."""
        save_dir = self._save_dir()
        key = f"org_{org.uid}_gen{org.generation}_f{org.food_eaten}_a{org.age}"
        pt_path = save_dir / f"{key}.pt"
        torch.save({
            "model_sd": {k: v.cpu() for k, v in org.model_sd.items()},
            "physical": {f.name: getattr(org.physical, f.name) for f in __import__('dataclasses').fields(org.physical)},
            "visual": {f.name: getattr(org.visual, f.name) for f in __import__('dataclasses').fields(org.visual)},
        }, str(pt_path))
        entry = {
            "key": key,
            "file": str(pt_path),
            "uid": org.uid,
            "lineage": org.lineage,
            "generation": org.generation,
            "food_eaten": org.food_eaten,
            "age": org.age,
            "note": note,
            "saved_at": int(time.time()),
            "physical": org.to_json()["physical"],
            "visual": org.to_json()["visual"],
        }
        idx = self._load_index()
        idx.append(entry)
        # Keep at most 200 saved organisms
        if len(idx) > 200:
            to_delete = idx[:-200]
            idx = idx[-200:]
            for old in to_delete:
                try:
                    Path(old["file"]).unlink(missing_ok=True)
                except Exception:
                    pass
        self._write_index(idx)
        return key

    def spawn_saved(self, key: str, position: np.ndarray | None = None) -> bool:
        """Spawn a saved organism into the current simulation. Returns True on success."""
        idx = self._load_index()
        entry = next((e for e in idx if e["key"] == key), None)
        if entry is None:
            return False
        pt_path = Path(entry["file"])
        if not pt_path.exists():
            return False
        data = torch.load(str(pt_path), map_location="cpu")
        physical = PhysicalTraits(**data["physical"])
        visual = VisualTraits(**data["visual"])
        model_sd = data["model_sd"]
        # Slightly mutate to avoid clones
        from .evo_genetics import mutate_weights, mutate_physical, mutate_visual
        model_sd = mutate_weights(model_sd, 0.005, self.rng)
        physical = mutate_physical(physical, self.rng, scale=0.3)
        visual = mutate_visual(visual, self.rng)
        uid = self._next_uid()
        org = self._make_organism(
            uid=uid,
            lineage=entry.get("lineage", uid % 12),
            generation=entry.get("generation", 0),
            sd=model_sd,
            physical=physical,
            visual=visual,
            position=position,
        )
        self.organisms.append(org)
        self.total_born += 1
        return True

    @staticmethod
    def list_saved() -> list[dict[str, Any]]:
        return EvolutionSimulation._load_index()
