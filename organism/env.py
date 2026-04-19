from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from .config import EnvironmentConfig


class Action(IntEnum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    EAT = 3
    REST = 4


ACTION_NAMES = {
    Action.FORWARD: "forward",
    Action.TURN_LEFT: "turn_left",
    Action.TURN_RIGHT: "turn_right",
    Action.EAT: "eat",
    Action.REST: "rest",
}


class ObsIndex(IntEnum):
    FOOD_LEFT = 0
    FOOD_CENTER = 1
    FOOD_RIGHT = 2
    HAZARD_LEFT = 3
    HAZARD_CENTER = 4
    HAZARD_RIGHT = 5
    WALL_LEFT = 6
    WALL_CENTER = 7
    WALL_RIGHT = 8
    SHELTER_ALIGNMENT = 9
    SHELTER_PROXIMITY = 10
    FOOD_CONTACT = 11
    SHELTER_CONTACT = 12
    ENERGY = 13
    DAMAGE = 14
    FATIGUE = 15
    NOVELTY_DRIVE = 16
    STRESS = 17


OBSERVATION_SIZE = len(ObsIndex)


@dataclass
class StepInfo:
    ate_food: bool
    hazard_contacts: int
    collision: bool
    reward: float
    discomfort: float
    energy: float
    damage: float
    fatigue: float
    novelty_bonus: float
    action_name: str
    death_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ate_food": self.ate_food,
            "hazard_contacts": self.hazard_contacts,
            "collision": self.collision,
            "reward": self.reward,
            "discomfort": self.discomfort,
            "energy": self.energy,
            "damage": self.damage,
            "fatigue": self.fatigue,
            "novelty_bonus": self.novelty_bonus,
            "action_name": self.action_name,
            "death_reason": self.death_reason,
        }


class OrganismEnv:
    """Small 2D world with homeostatic drives and deterministic resets."""

    def __init__(self, config: EnvironmentConfig, seed: int = 0) -> None:
        self.config = config
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.sector_offsets = np.array([-0.9, 0.0, 0.9], dtype=np.float32)
        self.food_positions = np.zeros((self.config.num_food_sources, 2), dtype=np.float32)
        self.food_cooldowns = np.zeros(self.config.num_food_sources, dtype=np.int32)
        self.hazard_positions = np.zeros((self.config.num_hazards, 2), dtype=np.float32)
        self.shelter_position = np.zeros(2, dtype=np.float32)
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.energy = 0.0
        self.damage = 0.0
        self.fatigue = 0.0
        self.episode_steps = 0
        self.episode_return = 0.0
        self.visitation = np.zeros(
            (self.config.novelty_grid_size, self.config.novelty_grid_size), dtype=np.int32
        )
        self.last_visit_cell: tuple[int, int] | None = None
        self.previous_discomfort = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.food_cooldowns.fill(0)
        self.visitation.fill(0)
        self.episode_steps = 0
        self.episode_return = 0.0
        self.last_visit_cell = None

        self.shelter_position[:] = np.array([0.5, 0.5], dtype=np.float32)
        self.agent_position[:] = self._spawn_near(self.shelter_position, radius=0.08)
        self.heading = float(self.rng.uniform(-np.pi, np.pi))
        self.energy = self.config.start_energy
        self.damage = self.config.start_damage
        self.fatigue = self.config.start_fatigue

        for index in range(self.config.num_food_sources):
            self.food_positions[index] = self._spawn_point(min_distance=0.15)

        for index in range(self.config.num_hazards):
            self.hazard_positions[index] = self._spawn_point(
                min_distance=0.18,
                avoid=np.vstack([self.food_positions, self.shelter_position[None, :]]),
            )

        self.previous_discomfort = self._discomfort()
        self._mark_visited()
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        chosen_action = Action(action)
        previous_discomfort = self._discomfort()
        ate_food = False
        collision = False

        self._tick_food_respawns()
        self.episode_steps += 1

        if chosen_action == Action.TURN_LEFT:
            self.heading -= self.config.turn_angle
            self.energy -= self.config.turn_energy_cost
            self.fatigue += 0.008
        elif chosen_action == Action.TURN_RIGHT:
            self.heading += self.config.turn_angle
            self.energy -= self.config.turn_energy_cost
            self.fatigue += 0.008
        elif chosen_action == Action.FORWARD:
            delta = np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32)
            next_position = self.agent_position + delta * self.config.move_speed
            clipped_position = np.clip(next_position, 0.0, self.config.world_size)
            collision = not np.allclose(clipped_position, next_position)
            self.agent_position[:] = clipped_position
            self.energy -= self.config.movement_energy_cost
            self.fatigue += 0.018
            if collision:
                self.damage += 0.02
        elif chosen_action == Action.EAT:
            edible_index = self._nearest_food_within(self.config.eat_radius)
            self.energy -= 0.002
            if edible_index is not None and self.food_cooldowns[edible_index] == 0:
                self.energy += self.config.food_energy_gain
                self.food_cooldowns[edible_index] = self.config.food_respawn_steps
                self.food_positions[edible_index] = np.array([-1.0, -1.0], dtype=np.float32)
                ate_food = True
            self.fatigue += 0.004
        elif chosen_action == Action.REST:
            shelter_distance = np.linalg.norm(self.agent_position - self.shelter_position)
            recovery_scale = 1.0 if shelter_distance <= self.config.shelter_radius else 0.35
            self.fatigue -= self.config.rest_recovery * recovery_scale
            self.damage -= self.config.passive_healing * recovery_scale
            self.energy += 0.006 * recovery_scale

        self.heading = float(((self.heading + np.pi) % (2 * np.pi)) - np.pi)
        self.energy -= self.config.energy_decay
        self.fatigue += 0.004

        hazard_contacts = self._apply_hazard_damage()
        if self._inside_shelter():
            self.damage -= self.config.passive_healing * 0.35
            self.fatigue -= 0.01

        novelty_bonus = self._mark_visited()
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        self.damage = float(np.clip(self.damage, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue, 0.0, 1.0))

        discomfort = self._discomfort()
        reward = (previous_discomfort - discomfort) + novelty_bonus
        if ate_food:
            reward += 0.2
        if collision:
            reward -= 0.05
        if hazard_contacts:
            reward -= 0.03 * hazard_contacts
        if chosen_action == Action.REST and self.fatigue < 0.15 and self.damage < 0.1:
            reward -= 0.01

        done = False
        death_reason: str | None = None
        if self.energy <= 0.0:
            done = True
            death_reason = "starvation"
            reward -= 0.5
        elif self.damage >= 1.0:
            done = True
            death_reason = "critical_damage"
            reward -= 0.5
        elif self.episode_steps >= self.config.max_steps:
            done = True

        self.previous_discomfort = discomfort
        self.episode_return += reward

        info = StepInfo(
            ate_food=ate_food,
            hazard_contacts=hazard_contacts,
            collision=collision,
            reward=reward,
            discomfort=discomfort,
            energy=self.energy,
            damage=self.damage,
            fatigue=self.fatigue,
            novelty_bonus=novelty_bonus,
            action_name=ACTION_NAMES[chosen_action],
            death_reason=death_reason,
        )
        return self._observe(), reward, done, info.to_dict()

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.episode_steps,
            "return": self.episode_return,
            "energy": self.energy,
            "damage": self.damage,
            "fatigue": self.fatigue,
        }

    def render_ascii(self, size: int = 14) -> str:
        grid = np.full((size, size), ".", dtype="<U1")
        shelter = self._to_grid(self.shelter_position, size)
        grid[shelter[1], shelter[0]] = "S"

        for food, cooldown in zip(self.food_positions, self.food_cooldowns):
            if cooldown == 0 and np.all(food >= 0.0):
                cell = self._to_grid(food, size)
                grid[cell[1], cell[0]] = "F"

        for hazard in self.hazard_positions:
            cell = self._to_grid(hazard, size)
            grid[cell[1], cell[0]] = "H"

        agent = self._to_grid(self.agent_position, size)
        grid[agent[1], agent[0]] = "A"
        return "\n".join("".join(row) for row in grid[::-1])

    def _observe(self) -> np.ndarray:
        available_food = self.food_positions[self.food_cooldowns == 0]
        food_sensors = self._sector_response(
            available_food, detection_range=self.config.food_visible_range
        )
        hazard_sensors = self._sector_response(self.hazard_positions)
        wall_sensors = np.array(
            [self._wall_sensor(self.heading + offset) for offset in self.sector_offsets],
            dtype=np.float32,
        )

        shelter_vector = self.shelter_position - self.agent_position
        shelter_distance = float(np.linalg.norm(shelter_vector))
        shelter_alignment = 0.5 * (
            np.cos(self._angle_to(shelter_vector) - self.heading) + 1.0
        )
        shelter_proximity = 1.0 - min(shelter_distance / self.config.sensor_range, 1.0)

        food_contact = 1.0 if self._nearest_food_within(self.config.eat_radius) is not None else 0.0
        shelter_contact = 1.0 if shelter_distance <= self.config.shelter_radius else 0.0
        current_visits = self._current_cell_visits()
        novelty_drive = 1.0 / (1.0 + current_visits)
        stress = np.clip(self._discomfort(), 0.0, 1.0)

        observation = np.concatenate(
            [
                food_sensors,
                hazard_sensors,
                wall_sensors,
                np.array(
                    [
                        shelter_alignment,
                        shelter_proximity,
                        food_contact,
                        shelter_contact,
                        self.energy,
                        self.damage,
                        self.fatigue,
                        novelty_drive,
                        stress,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return observation.astype(np.float32)

    def _tick_food_respawns(self) -> None:
        for index, cooldown in enumerate(self.food_cooldowns):
            if cooldown <= 0:
                continue
            self.food_cooldowns[index] -= 1
            if self.food_cooldowns[index] == 0:
                self.food_positions[index] = self._spawn_point(min_distance=0.15)

    def _apply_hazard_damage(self) -> int:
        contacts = 0
        for hazard in self.hazard_positions:
            distance = float(np.linalg.norm(self.agent_position - hazard))
            if distance <= self.config.hazard_radius:
                contacts += 1
                severity = 1.0 - (distance / self.config.hazard_radius)
                self.damage += self.config.hazard_damage * (0.45 + 0.55 * severity)
                self.energy -= 0.01 * severity
        return contacts

    def _mark_visited(self) -> float:
        x, y = self._visit_indices(self.agent_position)
        visits = self.visitation[y, x]
        self.visitation[y, x] += 1
        current_cell = (x, y)
        moved_to_new_cell = current_cell != self.last_visit_cell
        self.last_visit_cell = current_cell
        if not moved_to_new_cell:
            return 0.0
        return self.config.novelty_reward_scale / np.sqrt(1.0 + visits)

    def _current_cell_visits(self) -> float:
        x, y = self._visit_indices(self.agent_position)
        return float(self.visitation[y, x])

    def _visit_indices(self, point: np.ndarray) -> tuple[int, int]:
        grid_size = self.config.novelty_grid_size
        scaled = np.clip(point / self.config.world_size, 0.0, 0.9999)
        x = int(scaled[0] * grid_size)
        y = int(scaled[1] * grid_size)
        return x, y

    def _to_grid(self, point: np.ndarray, size: int) -> tuple[int, int]:
        clipped = np.clip(point / self.config.world_size, 0.0, 0.9999)
        x = int(clipped[0] * size)
        y = int(clipped[1] * size)
        return x, y

    def _inside_shelter(self) -> bool:
        return float(np.linalg.norm(self.agent_position - self.shelter_position)) <= self.config.shelter_radius

    def _nearest_food_within(self, radius: float) -> int | None:
        if not np.any(self.food_cooldowns == 0):
            return None
        available_indices = np.where(self.food_cooldowns == 0)[0]
        available_food = self.food_positions[available_indices]
        distances = np.linalg.norm(available_food - self.agent_position, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= radius:
            return int(available_indices[nearest])
        return None

    def _sector_response(
        self, targets: np.ndarray, detection_range: float | None = None
    ) -> np.ndarray:
        effective_range = detection_range if detection_range is not None else self.config.sensor_range
        response = np.zeros(3, dtype=np.float32)
        if len(targets) == 0:
            return response

        for target in targets:
            vector = target - self.agent_position
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-6 or distance > effective_range:
                continue

            relative_angle = self._wrap_angle(self._angle_to(vector) - self.heading)
            distance_weight = 1.0 - distance / effective_range
            for index, sector_center in enumerate(self.sector_offsets):
                angular_weight = max(0.0, np.cos(relative_angle - sector_center))
                response[index] += distance_weight * angular_weight

        return np.clip(response, 0.0, 1.0)

    def _wall_sensor(self, angle: float) -> float:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        distances = []
        for axis in range(2):
            component = direction[axis]
            if abs(component) < 1e-6:
                continue
            if component > 0.0:
                distance = (self.config.world_size - self.agent_position[axis]) / component
            else:
                distance = -self.agent_position[axis] / component
            if distance >= 0.0:
                distances.append(distance)

        ray_distance = min(distances) if distances else self.config.sensor_range
        normalized = 1.0 - min(ray_distance / self.config.sensor_range, 1.0)
        return float(np.clip(normalized, 0.0, 1.0))

    def _spawn_point(
        self,
        min_distance: float = 0.1,
        avoid: np.ndarray | None = None,
        max_attempts: int = 256,
    ) -> np.ndarray:
        for _ in range(max_attempts):
            point = self.rng.uniform(0.05, self.config.world_size - 0.05, size=2).astype(np.float32)
            if avoid is None:
                avoid_points = np.empty((0, 2), dtype=np.float32)
            else:
                avoid_points = avoid
            shelter_distance = np.linalg.norm(point - self.shelter_position)
            if shelter_distance < min_distance:
                continue
            if len(avoid_points):
                distances = np.linalg.norm(avoid_points - point, axis=1)
                if np.any(distances < min_distance):
                    continue
            return point
        return self.rng.uniform(0.05, self.config.world_size - 0.05, size=2).astype(np.float32)

    def _spawn_near(self, origin: np.ndarray, radius: float) -> np.ndarray:
        angle = self.rng.uniform(-np.pi, np.pi)
        offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return np.clip(origin + offset, 0.05, self.config.world_size - 0.05)

    def _discomfort(self) -> float:
        return float(
            np.clip(
                0.55 * (1.0 - self.energy) + 0.35 * self.damage + 0.25 * self.fatigue,
                0.0,
                1.5,
            )
        )

    @staticmethod
    def _angle_to(vector: np.ndarray) -> float:
        return float(np.arctan2(vector[1], vector[0]))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(((angle + np.pi) % (2 * np.pi)) - np.pi)
