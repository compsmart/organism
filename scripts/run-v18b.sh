#!/bin/bash
cd /root/organism
rm -rf outputs/organism-v18b
export PYTHONUNBUFFERED=1
python3 -u -m organism \
  --episodes 400 \
  --run-name organism-v18b \
  --seed 13 \
  --variety \
  --edge-penalty 0.01 \
  --food-approach 0.05 \
  --learning-rate 0.0005 \
  --entropy-coef 0.015 \
  --no-sleep \
  --no-reflex \
  --warm-start outputs/organism-v18/model_best.pt
