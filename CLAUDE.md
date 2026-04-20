# Organism Project — Claude Code Instructions

## Training Protocol

**Always use GPU on vast.ai for training. Never train locally on CPU.**

### Network constraint
The `vastai` Python CLI cannot reach `console.vast.ai` from this machine
(corporate proxy — Python's `requests` doesn't use Windows WinHTTP).
Use `scripts/vast.ps1` which calls the REST API via PowerShell's
`Invoke-RestMethod` (works through the Windows proxy stack).

### Before any training run

1. **Check for existing instances first** — never create a new one if one is already running:
   ```
   powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 list
   ```

2. **Reuse existing instance** if available. Current instance:
   - ID: 35272265 (A100 PCIE, $0.093/hr)
   - SSH: `ssh5.vast.ai:32264`
   - Image: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
   - IP: 193.205.162.51

3. **Show instance details** before connecting:
   ```
   powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 show 35272265
   ```

### Training workflow on existing instance

1. SSH into the instance (port + host from step 2 above)
2. Clone or pull the latest organism code
3. Install deps: `pip install torch numpy fastapi uvicorn pydantic`
4. Run training: `python -m organism --episodes 500 --run-name <name>`
5. Download the checkpoint via `scp`

### Only create new instances if

- No running instances exist
- Current instance has insufficient GPU for the task
- Instance has been destroyed

### Helper commands

```
powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 search       # find cheap offers
powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 list         # list my instances
powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 show <id>    # instance details
powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 create <id>  # create from offer
powershell -ExecutionPolicy Bypass -File scripts/vast.ps1 destroy <id> # destroy instance
```

## Autonomous Dev Loop

This project has an autonomous dev loop defined in the cron that reads
`.dev/state.json` and `.dev/progress.md` to resume. See the main dev
loop prompt for details. Always commit changes as you go.

## Code Conventions

- `organism/` — Python package
- `webui/` — Vanilla JS frontend (modules: app.js, panels.js, renderer.js)
- `.dev/state.json` — machine-readable state for autonomous loop
- `.dev/progress.md` — human-readable development log
- `.dev/synthesis.md` — accumulated findings across experiments
- Checkpoints saved to `outputs/<run-name>/model.pt`
