#!/bin/bash
# Autonomous training script for the organism project on vast.ai.
#
# What this does:
#   1. Installs deps (git, python3, torch, etc.)
#   2. Clones https://github.com/compsmart/organism
#   3. Runs training with CUDA if available
#   4. Uploads the checkpoint + metrics to 0x0.st (public pastebin)
#      and writes the URLs to a log file readable via the vast.ai API
#
# Configurable via env vars (set in instance create call):
#   ORGANISM_EPISODES   - number of training episodes (default 500)
#   ORGANISM_RUN_NAME   - output dir name (default gpu-run)
#   ORGANISM_SEED       - random seed (default 42)
#   ORGANISM_BRANCH     - git branch to clone (default main)

set -e

EPISODES="${ORGANISM_EPISODES:-500}"
RUN_NAME="${ORGANISM_RUN_NAME:-gpu-run}"
SEED="${ORGANISM_SEED:-42}"
BRANCH="${ORGANISM_BRANCH:-main}"
LOG=/root/organism-train.log

exec > >(tee "$LOG") 2>&1
echo "=========================================="
echo "Organism training starting at $(date)"
echo "Episodes: $EPISODES  Run: $RUN_NAME  Seed: $SEED  Branch: $BRANCH"
echo "=========================================="

# Install deps
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl python3-pip

# Clone code
cd /root
if [ ! -d organism ]; then
    git clone --depth 1 --branch "$BRANCH" https://github.com/compsmart/organism.git
fi
cd organism

# Install python deps. torch needs CUDA-matched version; the base nvidia image has cuda 12.1
pip install -q --upgrade pip
pip install -q torch numpy fastapi uvicorn pydantic

# Quick CUDA sanity check
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training
echo "=========================================="
echo "Training start: $(date)"
python -m organism --episodes "$EPISODES" --run-name "$RUN_NAME" --seed "$SEED"
echo "Training end: $(date)"
echo "=========================================="

OUT="outputs/$RUN_NAME"
if [ ! -f "$OUT/model.pt" ]; then
    echo "ERROR: checkpoint not found at $OUT/model.pt"
    exit 1
fi

# Upload checkpoint + metrics to 0x0.st (no auth, ~30-365 day retention)
echo "Uploading checkpoint..."
MODEL_URL=$(curl -sS -F "file=@$OUT/model.pt" https://0x0.st)
echo "Uploading metrics..."
METRICS_URL=$(curl -sS -F "file=@$OUT/metrics.jsonl" https://0x0.st)
EVALS_URL=$(curl -sS -F "file=@$OUT/evaluations.jsonl" https://0x0.st)

echo "=========================================="
echo "ORGANISM_UPLOAD_COMPLETE"
echo "MODEL_URL=$MODEL_URL"
echo "METRICS_URL=$METRICS_URL"
echo "EVALS_URL=$EVALS_URL"
echo "=========================================="

# Persist URLs in a known location so any API that can read instance state can find them
echo "$MODEL_URL" > /root/organism-model.url
echo "$METRICS_URL" > /root/organism-metrics.url
echo "$EVALS_URL" > /root/organism-evals.url

echo "All done at $(date). Instance can now be destroyed."
