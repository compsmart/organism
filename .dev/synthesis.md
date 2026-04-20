# Organism Development Synthesis

Accumulated findings from training experiments, ablations, and architecture decisions.

## Training Dynamics

### Episode Requirements
- **Minimum 300 episodes** needed for meaningful learning. Returns improve through ep 200-300.
- Early episodes (1-50): agent explores, begins finding food occasionally.
- Mid episodes (50-150): food-seeking behavior emerges, returns improve.
- Late episodes (150-300): policy stabilizes, best evals appear.
- **10-episode smoke tests** only verify no crashes, not learning quality.

### Survival Budget
- Start energy: 0.72, energy_decay: 0.005/step, movement cost: 0.01/step
- **Moving forward**: ~48 steps before starvation (no food)
- **Idle**: ~144 steps before starvation
- Agent must find and eat food within ~100 steps to sustain
- Food gives +0.38 energy — roughly 1 food per 50 steps sustains

### Food-Seeking Learning
- **eat_radius=0.12** (from 0.08): critical for last-mile navigation
- **food_reward_bonus=0.2** (from 0.08): strong enough learning signal
- **reflex EAT threshold=0.45** (from 0.24): bootstraps food-eating behavior
- **food_visible_range=0.3** (partial obs): forces memory use, validated via ablation

## Architecture Findings

### Stage 2: World Model + Surprise
- Surprise_coef=0.02 (tuned from 0.05): higher values destabilize policy
- Surprise initially drops (world model learns) then rises (policy changes)
- With sleep: last-half avg_return = -1.107
- Without sleep: last-half avg_return = -1.181
- Without surprise: last-half avg_return = -1.280
- **Conclusion**: surprise + sleep outperforms alternatives

### Stage 3: Episodic Memory (VALIDATED)
- Memory + partial obs: last-50 avg_return = **-0.564**
- No-memory + partial obs: last-50 avg_return = -0.704
- Full-vis + no memory: last-50 avg_return = -0.675
- **Memory provides 20% improvement** under partial observability
- Memory agent eats less food but makes better overall decisions (fewer penalties)

### Stage 4: Global Workspace
- Workspace vs no-workspace: equivalent performance (~-0.85 eval)
- **No measurable benefit** on current single-goal task
- Workspace creates infrastructure for multi-goal scenarios (future stages)
- **STABILITY ISSUE**: workspace broadcast causes policy collapse at ~ep100
  - Diagnosed: workspace is the culprit, not metacognition
  - No-workspace runs don't collapse; no-memory (with workspace) does collapse
  - **Fix applied**: detach workspace broadcast from gradient (0.3 * broadcast.detach())
- **Fix validated**: 300-ep run shows no collapse, first positive eval (+0.013), stable learning

### Stage 5: Self-Model
- sm_loss stable at 0.001-0.02 across training
- Body-state prediction (energy, damage, fatigue) works
- Ownership signal available but needs step-level analysis for full validation
- Architecturally complete — proper exercise comes with metacognition

### Stage 6: Metacognition
- Confidence head modulates logits: low confidence → flatter policy
- **STABILITY FIX**: detach confidence from policy gradient, narrow scale range (0.7 + 0.3c)
- Original (0.5 + 0.5c with gradient) contributed to degenerate policy

## Environment Tuning History

| Parameter | Original | Current | Reason |
|---|---|---|---|
| energy_decay | 0.012 | 0.005 | Agent starved too fast |
| sensor_range | 0.38 | 0.55 | Agent couldn't detect food |
| food_visible_range | - | 0.3 | Stage 3: partial observability |
| eat_radius | 0.08 | 0.12 | Last-mile navigation too hard |
| food_reward_bonus | 0.08 | 0.2 | Weak learning signal for eating |
| reflex EAT threshold | 0.24 | 0.45 | Reflex triggered too late |
| wall reflex threshold | 0.9 | 0.7 | Agent hit walls constantly |
| turn fatigue | 0.008 | 0.004 | Excessive fatigue from turning |
| rest outside shelter | 0.35x | 0.5x | Couldn't recover outside shelter |
| surprise_coef | 0.05 | 0.02 | Higher values destabilized policy |

## Stability Lessons

1. **Detach modulatory signals** from the main policy gradient. Workspace broadcast and confidence scaling both caused collapse when gradients flowed through them.
2. **Scale residual additions conservatively** (0.3x for workspace). Large residuals from new modules overwhelm the learned GRU representations.
3. **Test with 300 episodes minimum**. Collapse at ep100 is invisible in 10-episode smoke tests.
4. **Ablate systematically**: when full stack collapses, disable modules one at a time with the same seed to isolate the culprit.

## Lab Findings (Holonomic Track)

Key findings relevant to organism project:
- **D-2652** (5-star): Holonomic viable at d=256 (phase transition)
- **D-2654**: Cayley transform critical, gain gate dispensable
- **D-2658**: Zero forgetting with key-space separated domains
- **D-2660** (5-star): Forgetting is domain-structural, not architectural
- **D-2671**: N=16 ceiling is architectural, AMM may help
- **D-2682**: Overlap sweep confirms monotonic forgetting (rho=1.0)

**Implications for Stage 4a**: Holonomic recurrent core needs d=256 minimum, key-space separation for multi-behavior learning. AMM external memory **confirmed to solve N=16 ceiling** (D-2710, 5-star landmark).

### Stage 7: Dreaming
- dream_rollout() generates 16-step synthetic episodes via world model
- No environment reward in dreams — pure value bootstrapping
- dream_loss is small and decreasing after a few episodes
- Counterfactual simulation: agent imagines alternative strategies in latent space
