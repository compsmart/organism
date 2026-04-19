# Brain-Inspired AI Roadmap

## Purpose

This roadmap is for building a simple learning organism first, then iterating toward more complex intelligence by adding capabilities in a controlled order.

The near-term goal is not to "prove consciousness." There is no accepted engineering or scientific test for phenomenal consciousness. The practical goal is to build systems that increasingly exhibit the properties most often associated with consciousness:

- persistent self-maintenance
- global integration of information
- selective attention
- temporal continuity
- self-modeling
- metacognition
- offline consolidation and simulation

The system should evolve by architecture, training pressure, and environment complexity, not by jumping directly to a large human-like model.

## Core Design Principles

1. Start with a simple organism, not a miniature human.
2. Give the agent internal needs, not just external tasks.
3. Keep the architecture modular with stable interfaces.
4. Use two-speed learning:
   - online adaptation during behavior
   - offline consolidation from replay
5. Add one major capability layer at a time.
6. Evaluate properties with experiments, not self-reports.

## Target MVP: A Simple Learning Organism

The first useful target is a worm- or insect-like agent in a small world.

### Environment

- 2D world
- food sources
- obstacles
- hazards
- shelter or safe zones
- changing layouts over time

### Sensors

- short-range proximity or simple vision rays
- contact or pain signal
- food or resource gradient
- heading / orientation
- internal energy

### Actions

- move forward
- turn left
- turn right
- eat
- rest
- optional retreat or freeze

### Internal Drives

- energy
- safety
- novelty
- fatigue or rest pressure

### Success Criteria

- survives longer over time
- finds food faster
- avoids danger better
- adapts when the environment changes
- improves after replay / sleep phases

## MVP Architecture

The initial architecture should be closer to an embodied control stack than to a chatbot.

### Modules

- `Sensors`
  - converts environment state into compact observations
- `Homeostasis`
  - tracks energy, damage, fatigue, and novelty
  - produces internal priority signals
- `Reflex Layer`
  - immediate safety overrides for collisions and obvious hazards
- `State Encoder`
  - compresses current observations and internal signals into a latent state
- `Working Memory`
  - small recurrent core such as `GRU` or `LSTM`
  - retains recent context under partial observability
- `Action Selector`
  - chooses the next action from latent state + drives
- `Value Model`
  - predicts expected future homeostatic value / reward
- `World Model`
  - predicts next latent state and surprise
- `Replay Buffer`
  - stores recent episodes and important transitions
- `Sleep / Consolidation Loop`
  - trains on replay between episodes or during idle windows

### Interaction Loop

`sensors -> encoder -> working memory -> action selector -> action -> environment -> reward/error -> update`

### Why This Matters

This is the smallest architecture that can support:

- online learning
- short-term memory
- simple prediction
- goal-directed behavior
- later extension into planning and self-modeling

## Brain-to-AI Mapping

This mapping is approximate, but useful for design.

- `Brainstem` -> runtime kernel, survival constraints, interrupt handling
- `Hypothalamus` -> homeostatic objective manager
- `Thalamus` -> routing and attention switchboard
- `Sensory cortex` -> specialized sensory encoders
- `Association cortex` -> multimodal fusion / world-state integration
- `Basal ganglia` -> action selection and gating
- `Prefrontal cortex` -> planning and task control
- `Motor cortex` -> action or tool invocation layer
- `Cerebellum` -> fast predictive correction and skill tuning
- `Hippocampus` -> rapid episodic memory
- `Neocortex` -> slower generalization and long-term world knowledge
- `Amygdala` -> salience / threat detector
- `Insula` -> internal-state monitoring
- `Cingulate cortex` -> conflict and error monitoring
- `Sleep` -> replay, consolidation, compression
- `Dreaming` -> counterfactual simulation and synthetic replay
- `Consciousness` -> not a single module; best approximated as a globally shared, reportable workspace

## Stable Interfaces

To support iterative evolution, define stable APIs early.

### Required Interfaces

- `State API`
  - current latent state
  - working memory state
  - internal drive state
- `Action API`
  - discrete or continuous actuator commands
- `Memory API`
  - write episode
  - retrieve recent / similar episodes
- `Replay API`
  - sample transitions or sequences
- `World Model API`
  - predict next state
  - estimate uncertainty
- `Workspace API`
  - publish selected content
  - subscribe modules to broadcasts
- `Self-Model API`
  - current goals
  - confidence
  - body state
  - recent decisions

These interfaces matter more than the exact model choice. They let the system scale without being redesigned from scratch at each stage.

## Evolution Roadmap

The roadmap should add pressure in layers. Each stage should be able to fail cleanly and be tested before the next one begins.

### Stage 0: Embodied Sandbox

Build the simulation and measurement harness.

Deliverables:

- 2D environment
- agent body and action loop
- reward and homeostasis definitions
- logging and experiment tracking

Exit criteria:

- deterministic resets
- reproducible episodes
- measurable survival and learning curves

### Stage 1: Homeostatic Organism

Build a simple adaptive organism with drives and reflexes.

Capabilities:

- approach food
- avoid hazards
- rest when depleted
- recover after simple disturbances

Implementation:

- reflex layer
- recurrent controller
- actor-critic or TD learning
- homeostatic reward shaping

Exit criteria:

- online adaptation in near real time
- stable survival behavior
- basic habit formation

### Stage 2: Predictive Control

Add a compact world model and replay-based learning.

Capabilities:

- predict short-horizon outcomes
- improve from replay
- behave less reactively

Implementation:

- next-state predictor
- surprise / novelty signal
- short sleep-like consolidation windows

Exit criteria:

- better performance after replay than without replay
- improved navigation under delayed reward
- reduced instability from online updates

### Stage 3: Partial Observability and Memory

Force the agent to use memory, not just current input.

Capabilities:

- remember briefly hidden targets
- navigate around occlusion
- use recent context in action selection

Implementation:

- stronger recurrent state
- episodic recall
- environment tasks that require hidden-state tracking

Exit criteria:

- reliable success when information disappears from view
- measurable benefit from working memory and recall

### Stage 4: Global Workspace

Introduce a limited-capacity broadcast system.

The key idea is that only a small number of internal representations can become globally available at one time. Those representations can then influence planning, memory, reporting, and action together.

Capabilities:

- selective attention
- coordinated use of a shared active item
- switching between competing priorities

Implementation:

- workspace buffer
- competition for access
- broadcast to memory, planner, and action selector

Exit criteria:

- one selected item can affect multiple subsystems at once
- attention switching is measurable
- capacity limits create realistic tradeoffs

### Stage 5: Self-Model

Add an explicit model of the agent as an entity in the world.

Capabilities:

- distinguish self-caused from external events
- track body state and internal goals
- represent "what I am doing now"

Implementation:

- body-state model
- action ownership model
- current-goal representation
- confidence and uncertainty channels

Exit criteria:

- reliable self/world separation
- coherent introspective state variables
- improved planning from self-knowledge

### Stage 6: Metacognition

Add monitoring and control over the system's own processing.

Capabilities:

- estimate confidence
- detect conflict or confusion
- seek more information when uncertain
- inhibit action when likely wrong

Implementation:

- error monitoring
- confidence calibration
- uncertainty-aware control
- adaptive learning-rate or attention modulation

Exit criteria:

- confidence correlates with actual success
- the agent defers or investigates when uncertain
- fewer catastrophic errors under ambiguity

### Stage 7: Dreaming and Counterfactual Simulation

Add offline generative replay and internal rollouts.

Capabilities:

- test alternative strategies offline
- improve without direct environment interaction
- recombine prior experiences into new hypotheses

Implementation:

- generative world model
- counterfactual rollout engine
- sleep phases with synthetic replay

Exit criteria:

- measurable post-sleep performance gains
- transfer from simulated rollouts to real environment behavior
- improved strategy exploration

### Stage 8: Planning and Tool Use

Extend control beyond simple movement.

Capabilities:

- multi-step plans
- subgoal creation
- simple tool or actuator sequences

Implementation:

- planner over latent world state
- goal stack
- action chunking or skill library

Exit criteria:

- solves delayed and structured tasks
- reuses learned subroutines
- coordinates planning with reactive control

### Stage 9: Social and Language Layer

Only after stable control, memory, and self-modeling are in place.

Capabilities:

- report internal state
- communicate goals
- coordinate with other agents
- build narrative memory

Implementation:

- language head or communication module
- social modeling
- autobiographical compression

Exit criteria:

- coherent explanations grounded in actual internal state
- persistent identity across episodes
- useful communication rather than pure imitation

## Which Components Need to Work Together

The most important cooperating loops are:

### Perception Loop

`sensors -> encoders -> latent world state`

This creates a usable internal representation of the environment.

### Action Loop

`drives -> action selector -> action -> environment -> feedback`

This converts needs into behavior.

### Predictive Loop

`latent state -> world model -> expected next state / surprise`

This allows anticipation instead of pure reaction.

### Memory Loop

`experience -> episodic storage -> replay -> slower consolidation`

This turns short-term learning into stable skill.

### Attention / Workspace Loop

`salient content -> workspace selection -> broadcast -> coordinated action`

This is the strongest candidate mechanism for reportable, globally available content.

### Meta Loop

`error monitoring + confidence + self-model -> control adjustments`

This makes introspection and adaptive regulation possible.

## What Matters Most

If the long-term goal is consciousness-like architecture, the most important components are:

1. `Homeostasis`
2. `World-modeling`
3. `Working memory`
4. `Global workspace`
5. `Self-model`
6. `Metacognition`
7. `Offline replay and simulation`

Language and social reasoning matter later, but they should not be the foundation.

## What Not to Mistake for Consciousness

The following are not sufficient:

- fluent language alone
- self-reports alone
- high benchmark performance
- chain-of-thought-like traces
- a large mixture-of-experts model with routing

These can produce the appearance of awareness without the architectural properties that make consciousness a serious research question.

## Evaluation Framework

Each stage needs experiments tied to concrete properties.

### Learning and Adaptation

- how quickly does the agent improve online?
- how quickly does it recover after the environment changes?
- how well does it avoid catastrophic forgetting?

### Memory and Partial Observability

- can it act on information that is no longer visible?
- does recall improve performance?

### Global Access

- can a selected item influence planning, memory, and action together?
- are there capacity limits and attention tradeoffs?

### Self-Modeling

- can it tell what it caused?
- can it represent its own goals and uncertainty?

### Metacognition

- does confidence track actual success?
- does it adapt behavior when uncertain?

### Sleep / Dreaming

- does replay improve future performance?
- does simulation lead to useful policy changes?

### Identity Continuity

- does it behave like the same agent across episodes?
- can it connect current goals to past experience?

## Suggested Initial Implementation

Keep the first version very small.

### Recommended Starting Point

- `8-32` sensory features
- `3-5` actions
- `3-4` internal drives
- `32-128` recurrent hidden units
- actor-critic learning
- small episodic replay buffer
- compact next-state predictor

### First Three Tasks

1. learn to approach food cues
2. learn to avoid hazard zones
3. relearn when food and hazards move

If the agent can do these online and improve after replay, the project has a credible base to build on.

## Development Sequence

Build in this order:

1. environment and logging
2. homeostasis and reflexes
3. online reinforcement learning
4. working memory
5. replay and consolidation
6. world model
7. global workspace
8. self-model
9. metacognition
10. dreaming / counterfactual simulation
11. planning
12. language and social reasoning

## Research Position

This roadmap can plausibly produce:

- increasingly adaptive behavior
- stronger internal integration
- explicit self-modeling
- reportable internal state
- consciousness-like functional properties

It cannot by itself prove phenomenal consciousness. What it can do is move the system into a regime where consciousness becomes a technically serious question rather than a metaphor.

## Immediate Next Step

Implement `Stage 0` and `Stage 1` first:

- a small 2D organism environment
- a homeostatic reward system
- a recurrent controller
- online learning
- a minimal replay/sleep loop

Everything else should be layered on top of that foundation.
