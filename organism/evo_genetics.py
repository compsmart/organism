from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .evolution import PhysicalTraits, VisualTraits

PATTERNS = ["solid", "stripe", "spot", "ring"]
SHAPES = ["circle", "triangle", "diamond"]

# Per-trait Gaussian mutation sigmas
PHYSICAL_SIGMAS: dict[str, float] = {
    "sensor_range": 0.025,
    "food_visible_range": 0.030,
    "move_speed": 0.005,
    "turn_angle": 0.035,
    "fov_half_angle": 0.060,
    "energy_decay": 0.001,
    "eat_radius": 0.008,
    "hazard_sensitivity": 0.040,
}

PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sensor_range": (0.20, 0.90),
    "food_visible_range": (0.20, 1.00),
    "move_speed": (0.02, 0.12),
    "turn_angle": (0.15, 0.90),
    "fov_half_angle": (0.40, 1.40),
    "energy_decay": (0.002, 0.012),
    "eat_radius": (0.06, 0.20),
    "hazard_sensitivity": (0.70, 1.40),
}


def crossover_weights(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Per-parameter uniform crossover of two model state_dicts."""
    child: dict[str, torch.Tensor] = {}
    for key in sd_a:
        ta = sd_a[key].float()
        tb = sd_b[key].float()
        mask = torch.as_tensor(rng.integers(0, 2, size=ta.shape).astype(np.uint8), dtype=torch.bool)
        child[key] = torch.where(mask, ta, tb)
    return child


def mutate_weights(
    sd: dict[str, torch.Tensor],
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Add Gaussian noise to every parameter."""
    mutated: dict[str, torch.Tensor] = {}
    for key in sd:
        t = sd[key].float()
        noise = torch.as_tensor(
            rng.normal(0.0, sigma, size=t.shape).astype(np.float32)
        )
        mutated[key] = t + noise
    return mutated


def crossover_physical(pa: "PhysicalTraits", pb: "PhysicalTraits", rng: np.random.Generator) -> "PhysicalTraits":
    """Per-field uniform crossover — each field independently from one parent."""
    from .evolution import PhysicalTraits  # local import avoids circularity
    kwargs: dict[str, float] = {}
    for f in fields(pa):
        kwargs[f.name] = getattr(pa, f.name) if rng.integers(0, 2) == 0 else getattr(pb, f.name)
    return PhysicalTraits(**kwargs)


def mutate_physical(
    p: "PhysicalTraits",
    rng: np.random.Generator,
    scale: float = 1.0,
) -> "PhysicalTraits":
    """Perturb each numeric field by Gaussian noise, clip to bounds."""
    from .evolution import PhysicalTraits
    kwargs: dict[str, float] = {}
    for f in fields(p):
        v = getattr(p, f.name)
        sigma = PHYSICAL_SIGMAS.get(f.name, 0.0) * scale
        lo, hi = PHYSICAL_BOUNDS[f.name]
        v = float(np.clip(v + rng.normal(0.0, sigma), lo, hi))
        kwargs[f.name] = v
    return PhysicalTraits(**kwargs)


def crossover_visual(va: "VisualTraits", vb: "VisualTraits", rng: np.random.Generator) -> "VisualTraits":
    """Blend continuous visual traits; pick discrete ones from one parent."""
    from .evolution import VisualTraits

    def blend(a: float, b: float, noise_std: float) -> float:
        return float(a * 0.5 + b * 0.5 + rng.normal(0.0, noise_std))

    hue = blend(va.color_h, vb.color_h, 8.0) % 360.0
    sat = float(np.clip(blend(va.color_s, vb.color_s, 0.04), 0.3, 1.0))
    lit = float(np.clip(blend(va.color_l, vb.color_l, 0.04), 0.3, 0.8))
    size = float(np.clip(blend(va.body_size, vb.body_size, 0.05), 0.6, 1.8))
    pattern = va.pattern if rng.integers(0, 2) == 0 else vb.pattern
    shape = va.shape if rng.integers(0, 2) == 0 else vb.shape
    return VisualTraits(color_h=hue, color_s=sat, color_l=lit, pattern=pattern, body_size=size, shape=shape)


def mutate_visual(v: "VisualTraits", rng: np.random.Generator) -> "VisualTraits":
    """Small random mutation of visual traits; occasionally flip discrete ones."""
    from .evolution import VisualTraits

    hue = float((v.color_h + rng.normal(0.0, 5.0)) % 360.0)
    sat = float(np.clip(v.color_s + rng.normal(0.0, 0.03), 0.3, 1.0))
    lit = float(np.clip(v.color_l + rng.normal(0.0, 0.03), 0.3, 0.8))
    size = float(np.clip(v.body_size + rng.normal(0.0, 0.04), 0.6, 1.8))
    pattern = rng.choice(PATTERNS) if rng.random() < 0.05 else v.pattern  # type: ignore[arg-type]
    shape = rng.choice(SHAPES) if rng.random() < 0.05 else v.shape  # type: ignore[arg-type]
    return VisualTraits(color_h=hue, color_s=sat, color_l=lit, pattern=pattern, body_size=size, shape=shape)


def make_default_physical() -> "PhysicalTraits":
    from .evolution import PhysicalTraits
    return PhysicalTraits(
        sensor_range=0.55,
        food_visible_range=0.70,
        move_speed=0.06,
        turn_angle=0.40,
        fov_half_angle=0.90,
        energy_decay=0.005,
        eat_radius=0.12,
        hazard_sensitivity=1.00,
    )


def make_default_visual(lineage_index: int, total_lineages: int) -> "VisualTraits":
    from .evolution import VisualTraits
    hue = (lineage_index / max(total_lineages, 1)) * 360.0
    return VisualTraits(
        color_h=hue,
        color_s=0.75,
        color_l=0.55,
        pattern=PATTERNS[lineage_index % len(PATTERNS)],
        body_size=1.0,
        shape=SHAPES[lineage_index % len(SHAPES)],
    )


def seed_population(
    checkpoint_path: str | Path | None,
    n: int,
    initial_sigma: float,
    rng: np.random.Generator,
) -> list[tuple[dict[str, torch.Tensor], "PhysicalTraits", "VisualTraits"]]:
    """Return n (model_sd, physical, visual) tuples seeded from checkpoint with diversity."""
    base_sd: dict[str, torch.Tensor] | None = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            raw = torch.load(str(path), map_location="cpu")
            base_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    result = []
    for i in range(n):
        # Slightly increasing diversity for later organisms
        sigma = initial_sigma * (1.0 + 0.3 * i / max(n - 1, 1))
        if base_sd is not None:
            sd = mutate_weights(base_sd, sigma, rng)
        else:
            # No checkpoint — random weights
            from .agent import RecurrentActorCritic
            from .env import OBSERVATION_SIZE, Action
            model = RecurrentActorCritic(
                observation_size=OBSERVATION_SIZE,
                action_size=len(Action),
                hidden_size=256,
            )
            sd = {k: v.cpu() for k, v in model.state_dict().items()}

        phys = make_default_physical()
        phys = mutate_physical(phys, rng, scale=0.5 + 0.5 * i / max(n - 1, 1))
        vis = make_default_visual(i, n)
        result.append((sd, phys, vis))
    return result
