# Organism

Brain-inspired AI sandbox for building a simple embodied organism that can learn basic survival behavior in a 2D environment, then scale toward more complex architectures.

This repo currently includes:

- a small 2D organism environment with food, hazards, shelter, and homeostatic drives
- a recurrent actor-critic agent with simple reflex overrides
- replay-based "sleep" updates after each episode
- a local web UI for viewing and stepping the organism in the browser
- a desktop `tkinter` viewer
- a research roadmap for scaling from a simple organism toward richer intelligence

## Current Status

This is an early-stage MVP.

The agent is learning a small control problem:

- regulate `energy`, `damage`, and `fatigue`
- move through the world
- find food
- avoid hazards and walls
- rest when recovery helps

It is not yet a strong survivor, planner, or anything close to consciousness. The current implementation is meant to be a concrete foundation for iteration.

## Quickstart

Requirements:

- Python `3.11+`

Install dependencies:

```powershell
pip install -e .
```

Train a checkpoint:

```powershell
python -m organism.train --episodes 150 --run-name baseline
```

Launch the web UI:

```powershell
python -m organism.web
```

Then open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

Launch the desktop viewer:

```powershell
python -m organism.ui --checkpoint C:\projects\agents\organism\outputs\viewer_baseline\model.pt
```

## What The Agent Sees

The policy receives an observation vector containing:

- food direction: left, center, right
- hazard direction: left, center, right
- wall proximity: left, center, right
- shelter alignment and proximity
- food contact and shelter contact
- internal state: energy, damage, fatigue, novelty, stress

The available actions are:

- `forward`
- `turn_left`
- `turn_right`
- `eat`
- `rest`

## Learning Setup

The current agent is a small recurrent actor-critic:

- encoder -> `GRUCell` -> policy head + value head

Learning happens in two phases:

- online step-by-step TD updates during the episode
- replay-based "sleep" updates after the episode

There is also a simple reflex layer that can override obviously bad choices in a few survival-critical cases, such as eating when starving and food is already under the agent, or turning away from direct hazards.

## Web UI

The browser UI includes:

- responsive 2D world canvas
- checkpoint selector
- policy stepping and autoplay
- manual controls
- seed reset / randomize controls
- homeostasis meters
- directional sensor readouts
- movement trail and end-of-episode state

Keyboard shortcuts:

- `Space`: autoplay toggle
- `Enter`: policy step
- `ArrowUp`: forward
- `ArrowLeft`: turn left
- `ArrowRight`: turn right
- `E`: eat
- `R`: rest

## Repository Layout

- [organism/env.py](C:/projects/agents/organism/organism/env.py): environment and reward dynamics
- [organism/agent.py](C:/projects/agents/organism/organism/agent.py): recurrent policy, reflexes, replay, sleep updates
- [organism/session.py](C:/projects/agents/organism/organism/session.py): shared simulation session logic
- [organism/train.py](C:/projects/agents/organism/organism/train.py): training loop and checkpoint generation
- [organism/web.py](C:/projects/agents/organism/organism/web.py): FastAPI server for the web UI
- [webui/index.html](C:/projects/agents/organism/webui/index.html): browser UI shell
- [webui/app.js](C:/projects/agents/organism/webui/app.js): client rendering and controls
- [webui/styles.css](C:/projects/agents/organism/webui/styles.css): browser UI styles
- [ROADMAP.md](C:/projects/agents/organism/ROADMAP.md): staged research roadmap

## Outputs

Training runs are written under `outputs/<run-name>/` and typically include:

- `config.json`
- `metrics.jsonl`
- `evaluations.jsonl`
- `model.pt`
- `summary.json`

The `outputs/` directory is ignored by git.

## Roadmap

The next major steps are:

1. improve the reward design and policy stability so the organism survives longer
2. add a compact world model and surprise signal
3. move from reactive control toward predictive control
4. add stronger memory, then global workspace-style coordination
5. add explicit self-modeling and metacognitive signals later, not first

The longer-term architectural plan is documented in [ROADMAP.md](C:/projects/agents/organism/ROADMAP.md).

## Notes

- The current baseline checkpoints are useful for visualization, not as evidence of strong learning.
- The project is intentionally starting from a simple organism rather than a language-first architecture.
- Claims about consciousness are out of scope for the current codebase. The practical goal is to build the substrate for progressively richer control, memory, and self-modeling.
