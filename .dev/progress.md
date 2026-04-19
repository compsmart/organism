# Organism Development Progress

## Session Log

### 2026-04-19 — Stage 2 + Stage 3 Implementation

**Stage 2: Predictive Control (COMPLETE)**
- Implemented WorldModel (latent-space next-state predictor, ~5K params)
- Surprise signal augments reward (tuned coef=0.02)
- World model trains online + during sleep consolidation
- Ablation confirmed: surprise+sleep outperforms either alone
- Environment tuned: energy_decay 0.012→0.005, sensor_range 0.38→0.55
- Survival fixes: eat_radius 0.08→0.12, food bonus 0.08→0.2, reflex EAT threshold 0.24→0.45

**Stage 3: Partial Observability and Memory (IMPLEMENTED, VALIDATING)**
- Added food_visible_range=0.3 (food sensors shorter than hazard/wall)
- EpisodicMemory module: attention over last 16 hidden states, gated residual
- MemoryBuffer: sliding window managed per-episode
- Wall avoidance reflex threshold 0.9→0.7 (removed food-aware rule — too aggressive with partial obs)
- Training results: best eval -0.174, eating 1-4 food/episode, positive returns achieved
- Still needs ablation to validate exit criteria

**Roadmap Update**
- Added Stage 4a: Holonomic Recurrent Core based on lab track holonomic-sequence-model
- Key lab findings: D-2652 (viability at d=256), D-2654 (Cayley critical), D-2658 (zero forgetting)

## Pending Validation

- [ ] Stage 3 ablation: memory vs no-memory under partial observability
- [ ] Measure: does episodic memory improve food-finding under partial obs?
- [ ] Fatigue management: agent still exhausts from turning

### 2026-04-19 ~20:17 — Autonomous Loop: Lab Check + Ablation Launch

**Lab findings (2 new):**
- D-2660 (5-star): Key-space overlap IS causal for holonomic forgetting. Disjoint domains show positive transfer. Critical for Stage 4a key-space design.
- D-2659 (4-star): Holonomic O(1) state 2.88x memory advantage confirmed, but length generalization fails beyond training context.

**Action:** Launched 3 parallel ablation runs (300 eps each, seed=7):
1. `ablation-memory` — partial obs + episodic memory (full Stage 3)
2. `ablation-no-memory` — partial obs + no memory
3. `ablation-full-vis` — full visibility + no memory (Stage 2 baseline)

Results pending — will analyze on next invocation.
