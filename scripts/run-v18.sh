#!/bin/bash
cd /root/organism
rm -rf outputs/organism-v18
export PYTHONUNBUFFERED=1
python3 -u -m organism \
  --episodes 600 \
  --run-name organism-v18 \
  --seed 7 \
  --variety \
  --edge-penalty 0.01 \
  --food-approach 0.05 \
  --learning-rate 0.001 \
  --entropy-coef 0.02 \
  --no-sleep \
  --no-reflex \
  --warm-start outputs/organism-v17/model_best.pt
