from __future__ import annotations

import argparse
import math
import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from .env import Action, ObsIndex
from .session import SimulationSession, load_config


CANVAS_SIZE = 640
PANEL_WIDTH = 320

class OrganismViewer:
    def __init__(
        self,
        session: SimulationSession,
        autoplay_ms: int,
    ) -> None:
        self.session = session
        self.autoplay_ms = autoplay_ms
        self.root = tk.Tk()
        self.root.title("Organism Viewer")
        self.root.geometry(f"{CANVAS_SIZE + PANEL_WIDTH}x{CANVAS_SIZE + 24}")
        self.root.minsize(CANVAS_SIZE + PANEL_WIDTH, CANVAS_SIZE)

        self.autoplay = False
        self.show_sensors = tk.BooleanVar(value=True)
        self.deterministic_var = tk.BooleanVar(value=session.deterministic)
        self.status_var = tk.StringVar(value="Ready")
        self.seed_var = tk.StringVar(value=str(session.seed))
        self.autoplay_label_var = tk.StringVar(value="Autoplay: off")

        self.metric_vars = {
            "step": tk.StringVar(),
            "return": tk.StringVar(),
            "energy": tk.StringVar(),
            "damage": tk.StringVar(),
            "fatigue": tk.StringVar(),
            "stress": tk.StringVar(),
            "action": tk.StringVar(),
            "reward": tk.StringVar(),
            "reflex": tk.StringVar(),
        }
        self.sensor_vars = {
            "food": tk.StringVar(),
            "hazard": tk.StringVar(),
            "wall": tk.StringVar(),
            "shelter": tk.StringVar(),
            "internal": tk.StringVar(),
        }

        self._build_layout()
        self._bind_inputs()
        self.refresh()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=0)
        container.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            container,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="#f4f5ef",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        panel = ttk.Frame(container, width=PANEL_WIDTH)
        panel.grid(row=0, column=1, sticky="ns")

        controls = ttk.LabelFrame(panel, text="Controls", padding=10)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Policy Step", command=self.step_policy).pack(fill=tk.X)
        ttk.Button(controls, text="Toggle Autoplay", command=self.toggle_autoplay).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Reset Episode", command=self.reset_episode).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls, text="Random Seed", command=self.randomize_seed).pack(fill=tk.X, pady=(6, 0))

        manual = ttk.Frame(controls)
        manual.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(manual, text="Forward", command=lambda: self.step_manual(Action.FORWARD)).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(manual, text="Turn Left", command=lambda: self.step_manual(Action.TURN_LEFT)).grid(
            row=1, column=0, sticky="ew", pady=(4, 0)
        )
        ttk.Button(manual, text="Turn Right", command=lambda: self.step_manual(Action.TURN_RIGHT)).grid(
            row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0)
        )
        ttk.Button(manual, text="Eat", command=lambda: self.step_manual(Action.EAT)).grid(
            row=2, column=0, sticky="ew", pady=(4, 0)
        )
        ttk.Button(manual, text="Rest", command=lambda: self.step_manual(Action.REST)).grid(
            row=2, column=1, sticky="ew", padx=(4, 0), pady=(4, 0)
        )
        manual.columnconfigure(0, weight=1)
        manual.columnconfigure(1, weight=1)

        options = ttk.Frame(controls)
        options.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="Show Sensors",
            variable=self.show_sensors,
            command=self.refresh,
        ).pack(anchor="w")
        ttk.Checkbutton(
            options,
            text="Deterministic Policy",
            variable=self.deterministic_var,
            command=self.on_toggle_deterministic,
        ).pack(anchor="w")
        ttk.Label(options, textvariable=self.autoplay_label_var).pack(anchor="w", pady=(6, 0))

        seed_row = ttk.Frame(controls)
        seed_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(seed_row, text="Seed").pack(side=tk.LEFT)
        ttk.Entry(seed_row, textvariable=self.seed_var, width=10).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(seed_row, text="Apply", command=self.apply_seed).pack(side=tk.RIGHT)

        metrics = ttk.LabelFrame(panel, text="State", padding=10)
        metrics.pack(fill=tk.X, pady=(12, 0))
        for label, key in [
            ("Step", "step"),
            ("Return", "return"),
            ("Energy", "energy"),
            ("Damage", "damage"),
            ("Fatigue", "fatigue"),
            ("Stress", "stress"),
            ("Action", "action"),
            ("Reward", "reward"),
            ("Reflex", "reflex"),
        ]:
            row = ttk.Frame(metrics)
            row.pack(fill=tk.X)
            ttk.Label(row, text=f"{label}:", width=10).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=self.metric_vars[key]).pack(side=tk.LEFT)

        sensors = ttk.LabelFrame(panel, text="Sensors", padding=10)
        sensors.pack(fill=tk.X, pady=(12, 0))
        for label, key in [
            ("Food", "food"),
            ("Hazard", "hazard"),
            ("Wall", "wall"),
            ("Shelter", "shelter"),
            ("Internal", "internal"),
        ]:
            row = ttk.Frame(sensors)
            row.pack(fill=tk.X)
            ttk.Label(row, text=f"{label}:", width=10).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=self.sensor_vars[key], wraplength=220, justify=tk.LEFT).pack(
                side=tk.LEFT
            )

        status = ttk.LabelFrame(panel, text="Status", padding=10)
        status.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        ttk.Label(status, textvariable=self.status_var, wraplength=260, justify=tk.LEFT).pack(anchor="w")

    def _bind_inputs(self) -> None:
        self.root.bind("<space>", lambda _event: self.toggle_autoplay())
        self.root.bind("<Return>", lambda _event: self.step_policy())
        self.root.bind("w", lambda _event: self.step_manual(Action.FORWARD))
        self.root.bind("a", lambda _event: self.step_manual(Action.TURN_LEFT))
        self.root.bind("d", lambda _event: self.step_manual(Action.TURN_RIGHT))
        self.root.bind("e", lambda _event: self.step_manual(Action.EAT))
        self.root.bind("r", lambda _event: self.step_manual(Action.REST))
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

    def on_toggle_deterministic(self) -> None:
        self.session.deterministic = self.deterministic_var.get()
        self.refresh()

    def apply_seed(self) -> None:
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            self.status_var.set("Seed must be an integer.")
            return
        self.session.reset(seed=seed)
        self.status_var.set(f"Episode reset with seed {seed}.")
        self.refresh()

    def randomize_seed(self) -> None:
        seed = random.randint(0, 1_000_000)
        self.seed_var.set(str(seed))
        self.session.reset(seed=seed)
        self.status_var.set(f"Episode reset with random seed {seed}.")
        self.refresh()

    def toggle_autoplay(self) -> None:
        self.autoplay = not self.autoplay
        self.autoplay_label_var.set(f"Autoplay: {'on' if self.autoplay else 'off'}")
        if self.autoplay:
            self.status_var.set("Autoplay enabled.")
            self._schedule_tick()
        else:
            self.status_var.set("Autoplay paused.")

    def reset_episode(self) -> None:
        self.session.reset(seed=self.session.seed)
        self.autoplay = False
        self.autoplay_label_var.set("Autoplay: off")
        self.status_var.set(f"Episode reset with seed {self.session.seed}.")
        self.refresh()

    def step_policy(self) -> None:
        if self.session.learner is None:
            self.status_var.set("Load a checkpoint to use policy stepping.")
            return
        self._step(lambda: self.session.step_policy())

    def step_manual(self, action: Action) -> None:
        self._step(lambda: self.session.step_manual(action))

    def _step(self, step_fn) -> None:
        if self.session.last_step.done:
            self.status_var.set("Episode has ended. Reset to continue.")
            self.autoplay = False
            self.autoplay_label_var.set("Autoplay: off")
            return

        step_state = step_fn()
        if step_state.done:
            if step_state.death_reason:
                self.status_var.set(f"Episode ended: {step_state.death_reason}.")
            else:
                self.status_var.set("Episode reached max steps.")
            self.autoplay = False
            self.autoplay_label_var.set("Autoplay: off")
        else:
            mode = "policy" if self.session.learner is not None else "manual"
            self.status_var.set(
                f"Stepped via {mode}: {step_state.action_name}, reward {step_state.reward:+.3f}."
            )
        self.refresh()

    def _schedule_tick(self) -> None:
        if not self.autoplay:
            return
        if self.session.last_step.done:
            self.autoplay = False
            self.autoplay_label_var.set("Autoplay: off")
            return
        if self.session.learner is None:
            self.autoplay = False
            self.autoplay_label_var.set("Autoplay: off")
            self.status_var.set("Autoplay requires a checkpoint.")
            return
        self.step_policy()
        self.root.after(self.autoplay_ms, self._schedule_tick)

    def refresh(self) -> None:
        self._draw_world()
        self._update_metrics()

    def _draw_world(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, CANVAS_SIZE, CANVAS_SIZE, fill="#f4f5ef", outline="")

        grid_size = self.session.env.config.novelty_grid_size
        cell = CANVAS_SIZE / grid_size
        max_visits = max(float(np.max(self.session.env.visitation)), 1.0)
        for y in range(grid_size):
            for x in range(grid_size):
                visits = self.session.env.visitation[y, x]
                if visits <= 0:
                    continue
                alpha = min(visits / max_visits, 1.0)
                shade = int(244 - alpha * 42)
                color = f"#{shade:02x}{shade:02x}{228:02x}"
                x0 = x * cell
                y0 = CANVAS_SIZE - (y + 1) * cell
                self.canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=color, outline="")

        self._draw_shelter()
        self._draw_food()
        self._draw_hazards()
        self._draw_trail()
        if self.show_sensors.get():
            self._draw_sensor_rays()
        self._draw_agent()
        self.canvas.create_rectangle(1, 1, CANVAS_SIZE - 1, CANVAS_SIZE - 1, outline="#4a4d43", width=2)

    def _draw_shelter(self) -> None:
        env = self.session.env
        x, y = self._world_to_canvas(env.shelter_position)
        radius = env.config.shelter_radius * CANVAS_SIZE
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="#d9efe0",
            outline="#688b73",
            width=2,
        )
        self.canvas.create_text(x, y, text="S", fill="#355342", font=("Segoe UI", 12, "bold"))

    def _draw_food(self) -> None:
        env = self.session.env
        radius = env.config.eat_radius * CANVAS_SIZE * 0.75
        for position, cooldown in zip(env.food_positions, env.food_cooldowns):
            if cooldown > 0 or np.any(position < 0.0):
                continue
            x, y = self._world_to_canvas(position)
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#5ca35c",
                outline="#2e6b2e",
                width=2,
            )

    def _draw_hazards(self) -> None:
        env = self.session.env
        radius = env.config.hazard_radius * CANVAS_SIZE
        for position in env.hazard_positions:
            x, y = self._world_to_canvas(position)
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#f1b0a9",
                outline="#9a3f38",
                width=2,
            )
            self.canvas.create_text(x, y, text="!", fill="#7b211b", font=("Segoe UI", 11, "bold"))

    def _draw_agent(self) -> None:
        env = self.session.env
        x, y = self._world_to_canvas(env.agent_position)
        radius = 10
        fill = "#2d516d" if not self.session.last_step.done else "#6a6f77"
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="#112535", width=2)

        arrow_length = 24
        dx = math.cos(env.heading) * arrow_length
        dy = -math.sin(env.heading) * arrow_length
        self.canvas.create_line(x, y, x + dx, y + dy, fill="#0a1620", width=3, arrow=tk.LAST)

        label = self.session.last_step.action_name
        if self.session.last_step.reflex_override:
            label = f"{label} (reflex)"
        self.canvas.create_text(x, y - 18, text=label, fill="#0a1620", font=("Segoe UI", 10, "bold"))

    def _draw_trail(self) -> None:
        if len(self.session.position_history) < 2:
            return
        points = []
        start = max(0, len(self.session.position_history) - 60)
        history = self.session.position_history[start:]
        for point in history:
            x, y = self._world_to_canvas(point)
            points.extend([x, y])
        self.canvas.create_line(*points, fill="#5a7892", width=2, smooth=True)
        for index, point in enumerate(history[:-1]):
            if index % 6 != 0:
                continue
            x, y = self._world_to_canvas(point)
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#5a7892", outline="")

    def _draw_sensor_rays(self) -> None:
        env = self.session.env
        origin_x, origin_y = self._world_to_canvas(env.agent_position)
        reach = env.config.sensor_range * CANVAS_SIZE
        sensor_sets = [
            (
                [ObsIndex.FOOD_LEFT, ObsIndex.FOOD_CENTER, ObsIndex.FOOD_RIGHT],
                "#4f9f4f",
            ),
            (
                [ObsIndex.HAZARD_LEFT, ObsIndex.HAZARD_CENTER, ObsIndex.HAZARD_RIGHT],
                "#b44f4f",
            ),
            (
                [ObsIndex.WALL_LEFT, ObsIndex.WALL_CENTER, ObsIndex.WALL_RIGHT],
                "#7f7f7f",
            ),
        ]
        for indices, color in sensor_sets:
            for offset, index in zip(env.sector_offsets, indices):
                magnitude = float(self.session.observation[index])
                angle = env.heading + float(offset)
                end_x = origin_x + math.cos(angle) * reach * magnitude
                end_y = origin_y - math.sin(angle) * reach * magnitude
                self.canvas.create_line(origin_x, origin_y, end_x, end_y, fill=color, width=2, dash=(4, 3))

    def _update_metrics(self) -> None:
        env = self.session.env
        observation = self.session.observation
        stress = env._discomfort()
        self.metric_vars["step"].set(str(env.episode_steps))
        self.metric_vars["return"].set(f"{env.episode_return:+.3f}")
        self.metric_vars["energy"].set(f"{env.energy:.3f}")
        self.metric_vars["damage"].set(f"{env.damage:.3f}")
        self.metric_vars["fatigue"].set(f"{env.fatigue:.3f}")
        self.metric_vars["stress"].set(f"{stress:.3f}")
        self.metric_vars["action"].set(self.session.last_step.action_name)
        self.metric_vars["reward"].set(f"{self.session.last_step.reward:+.3f}")
        self.metric_vars["reflex"].set("yes" if self.session.last_step.reflex_override else "no")

        self.sensor_vars["food"].set(
            self._format_triplet(
                observation[ObsIndex.FOOD_LEFT],
                observation[ObsIndex.FOOD_CENTER],
                observation[ObsIndex.FOOD_RIGHT],
            )
        )
        self.sensor_vars["hazard"].set(
            self._format_triplet(
                observation[ObsIndex.HAZARD_LEFT],
                observation[ObsIndex.HAZARD_CENTER],
                observation[ObsIndex.HAZARD_RIGHT],
            )
        )
        self.sensor_vars["wall"].set(
            self._format_triplet(
                observation[ObsIndex.WALL_LEFT],
                observation[ObsIndex.WALL_CENTER],
                observation[ObsIndex.WALL_RIGHT],
            )
        )
        self.sensor_vars["shelter"].set(
            f"align={observation[ObsIndex.SHELTER_ALIGNMENT]:.2f}, prox={observation[ObsIndex.SHELTER_PROXIMITY]:.2f}, contact={observation[ObsIndex.SHELTER_CONTACT]:.0f}"
        )
        self.sensor_vars["internal"].set(
            f"energy={observation[ObsIndex.ENERGY]:.2f}, damage={observation[ObsIndex.DAMAGE]:.2f}, fatigue={observation[ObsIndex.FATIGUE]:.2f}, novelty={observation[ObsIndex.NOVELTY_DRIVE]:.2f}"
        )

    @staticmethod
    def _format_triplet(left: float, center: float, right: float) -> str:
        return f"L={left:.2f}, C={center:.2f}, R={right:.2f}"

    @staticmethod
    def _world_to_canvas(point: np.ndarray) -> tuple[float, float]:
        x = float(point[0]) * CANVAS_SIZE
        y = CANVAS_SIZE - float(point[1]) * CANVAS_SIZE
        return x, y

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the organism 2D viewer.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to the saved config.json for the checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Episode seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force greedy policy actions. Default viewer mode is stochastic for more visible behavior.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Deprecated alias; stochastic playback is already the default.",
    )
    parser.add_argument(
        "--autoplay-ms",
        type=int,
        default=120,
        help="Delay in milliseconds between autoplay steps.",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Construct and destroy the window immediately to validate the UI entrypoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.checkpoint)
    session = SimulationSession(
        config=config,
        checkpoint=args.checkpoint,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    viewer = OrganismViewer(session=session, autoplay_ms=args.autoplay_ms)
    if args.headless_check:
        viewer.root.update_idletasks()
        viewer.root.destroy()
        print("viewer_ok")
        return
    viewer.run()


if __name__ == "__main__":
    main()
