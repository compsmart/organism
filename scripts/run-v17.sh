#!/bin/bash
cd /root/organism
rm -rf outputs/organism-v17
export PYTHONUNBUFFERED=1
python3 -u -m organism \
  --episodes 700 \
  --run-name organism-v17 \
  --seed 42 \
  --variety \
  --edge-penalty 0.01 \
  --food-approach 0.05 \
  --learning-rate 0.0015 \
  --entropy-coef 0.03 \
  --no-sleep \
  --no-reflex
