from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvironmentConfig:
    world_size: float = 1.0
    max_steps: int = 256
    num_food_sources: int = 5
    num_hazards: int = 5
    sensor_range: float = 0.55
    eat_radius: float = 0.12
    shelter_radius: float = 0.12
    move_speed: float = 0.06
    turn_angle: float = 0.4
    energy_decay: float = 0.005
    movement_energy_cost: float = 0.008
    turn_energy_cost: float = 0.004
    food_energy_gain: float = 0.5
    hazard_radius: float = 0.1
    hazard_damage: float = 0.22
    rest_recovery: float = 0.08
    passive_healing: float = 0.012
    novelty_grid_size: int = 8
    novelty_reward_scale: float = 0.0
    food_respawn_steps: int = 10
    start_energy: float = 0.72
    start_damage: float = 0.0
    start_fatigue: float = 0.12
    food_visible_range: float = 0.7
    edge_hazard_curriculum: bool = False
    edge_penalty: float = 0.0
    edge_threshold: float = 0.1
    food_approach_scale: float = 0.0
    variety: bool = False
    food_quantity_min: float = 2.0
    food_quantity_max: float = 6.0
    eat_rate: float = 0.5
    hazard_radius_min: float = 0.06
    hazard_radius_max: float = 0.18
    hazard_speed: float = 0.003


@dataclass
class AgentConfig:
    hidden_size: int = 256
    reference_hidden_size: int = 64
    learning_rate: float = 3e-3
    gamma: float = 0.97
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    grad_clip: float = 1.0
    wm_learning_rate: float = 1e-3
    surprise_coef: float = 0.0
    max_surprise_bonus: float = 0.1
    episodic_memory_slots: int = 16
    use_episodic_memory: bool = True
    use_global_workspace: bool = True
    use_metacognition: bool = True
    use_planning: bool = True
    use_narration: bool = True
    use_reflex: bool = True


@dataclass
class EvolutionConfig:
    population_size: int = 12
    max_population: int = 24
    min_population: int = 4
    min_mate_age: int = 120
    mate_energy_min: float = 0.55
    mate_damage_max: float = 0.35
    mate_fatigue_max: float = 0.65
    mate_food_min: int = 5
    mate_radius: float = 0.08
    hatch_countdown: int = 40
    mutation_sigma: float = 0.015
    trait_mutation_scale: float = 1.0
    initial_diversity_sigma: float = 0.01
    seed_checkpoint: str = ""
    mate_approach_scale: float = 0.04


@dataclass
class TrainingConfig:
    seed: int = 7
    episodes: int = 500
    log_every: int = 10
    eval_every: int = 10
    eval_episodes: int = 10
    replay_capacity: int = 64
    min_replay_episodes: int = 4
    sleep_batches: int = 4
    sleep_batch_size: int = 8
    sleep_seq_len: int = 16
    sleep_burn_in: int = 6
    output_dir: Path = Path("outputs")
    run_name: str = "baseline"


@dataclass
class ExperimentConfig:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        env = EnvironmentConfig(**payload.get("env", {}))
        agent = AgentConfig(**payload.get("agent", {}))
        train_payload = dict(payload.get("train", {}))
        if "output_dir" in train_payload:
            train_payload["output_dir"] = Path(train_payload["output_dir"])
        train = TrainingConfig(**train_payload)
        return cls(env=env, agent=agent, train=train)


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value
