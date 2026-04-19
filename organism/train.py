from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import EpisodeRecord, OrganismLearner
from .config import ExperimentConfig
from .env import ACTION_NAMES, OBSERVATION_SIZE, Action, OrganismEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the brain-inspired organism MVP.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the run.")
    parser.add_argument("--run-name", type=str, default=None, help="Output subdirectory for the run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used for logs, config, and checkpoints.",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Print an ASCII snapshot from the first evaluation episode.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize_episode(
    episode_index: int,
    env: OrganismEnv,
    action_counts: dict[str, int],
    reflex_overrides: int,
    update_stats: dict[str, float],
    sleep_stats: dict[str, float],
    death_reason: str | None,
) -> dict[str, Any]:
    summary = env.summary()
    summary.update(
        {
            "episode": episode_index,
            "action_counts": action_counts,
            "reflex_overrides": reflex_overrides,
            "online_update": update_stats,
            "sleep_update": sleep_stats,
            "death_reason": death_reason,
        }
    )
    return summary


def run_evaluation(
    config: ExperimentConfig,
    learner: OrganismLearner,
    render: bool,
    base_seed: int,
) -> dict[str, Any]:
    eval_env = OrganismEnv(config.env, seed=base_seed)
    returns = []
    steps = []
    renders = []

    for episode_index in range(config.train.eval_episodes):
        observation = eval_env.reset(seed=base_seed + episode_index)
        hidden = learner.initial_hidden()
        done = False
        while not done:
            step = learner.select_action(observation, hidden, deterministic=True)
            observation, _, done, _ = eval_env.step(step.action)
            hidden = step.hidden.detach()
        summary = eval_env.summary()
        returns.append(summary["return"])
        steps.append(summary["steps"])
        if render and episode_index == 0:
            renders.append(eval_env.render_ascii())

    evaluation = {
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps)),
    }
    if renders:
        evaluation["render"] = renders[0]
    return evaluation


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def build_experiment(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    if args.episodes is not None:
        config.train.episodes = args.episodes
    if args.seed is not None:
        config.train.seed = args.seed
    if args.run_name is not None:
        config.train.run_name = args.run_name
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    return config


def main() -> None:
    args = parse_args()
    config = build_experiment(args)
    set_global_seed(config.train.seed)

    run_root = config.train.output_dir / config.train.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    metrics_path = run_root / "metrics.jsonl"
    evaluations_path = run_root / "evaluations.jsonl"
    checkpoint_path = run_root / "model.pt"

    write_json(
        run_root / "config.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config": config.to_dict(),
        },
    )

    env = OrganismEnv(config.env, seed=config.train.seed)
    learner = OrganismLearner(
        observation_size=OBSERVATION_SIZE,
        action_size=len(Action),
        agent_config=config.agent,
        training_config=config.train,
        seed=config.train.seed,
    )

    for episode_index in range(1, config.train.episodes + 1):
        observation = env.reset(seed=config.train.seed + episode_index)
        hidden = learner.initial_hidden()
        episode = EpisodeRecord(observations=[observation.copy()], actions=[], rewards=[], dones=[])
        done = False
        action_counts = {name: 0 for name in ACTION_NAMES.values()}
        reflex_overrides = 0
        death_reason: str | None = None
        last_update_stats: dict[str, float] = {}

        while not done:
            step = learner.select_action(observation, hidden)
            next_observation, reward, done, info = env.step(step.action)
            next_value = learner.bootstrap_value(next_observation, step.hidden.detach())
            last_update_stats = learner.online_update(step, reward, done, next_value)

            observation = next_observation
            hidden = step.hidden.detach()
            episode.observations.append(observation.copy())
            episode.actions.append(step.action)
            episode.rewards.append(reward)
            episode.dones.append(done)
            action_counts[info["action_name"]] += 1
            reflex_overrides += int(step.reflex_override)
            death_reason = info["death_reason"]

        learner.replay.add_episode(episode)
        sleep_stats = learner.sleep_update()
        summary = summarize_episode(
            episode_index=episode_index,
            env=env,
            action_counts=action_counts,
            reflex_overrides=reflex_overrides,
            update_stats=last_update_stats,
            sleep_stats=sleep_stats,
            death_reason=death_reason,
        )
        append_jsonl(metrics_path, summary)

        if episode_index % config.train.log_every == 0 or episode_index == 1:
            print(
                (
                    f"episode={episode_index:04d} "
                    f"return={summary['return']:.3f} "
                    f"steps={summary['steps']:03d} "
                    f"energy={summary['energy']:.3f} "
                    f"damage={summary['damage']:.3f} "
                    f"fatigue={summary['fatigue']:.3f} "
                    f"reflex={summary['reflex_overrides']}"
                )
            )

        if episode_index % config.train.eval_every == 0:
            evaluation = run_evaluation(
                config=config,
                learner=learner,
                render=args.render_eval,
                base_seed=config.train.seed + 10_000 + episode_index,
            )
            evaluation["episode"] = episode_index
            append_jsonl(evaluations_path, evaluation)
            print(
                (
                    f"eval@{episode_index:04d} "
                    f"avg_return={evaluation['avg_return']:.3f} "
                    f"avg_steps={evaluation['avg_steps']:.1f}"
                )
            )
            if args.render_eval and "render" in evaluation:
                print(evaluation["render"])

    learner.save(str(checkpoint_path))
    final_summary = {
        "episodes": config.train.episodes,
        "checkpoint": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "evaluations_path": str(evaluations_path),
    }
    write_json(run_root / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
