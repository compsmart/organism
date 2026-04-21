#!/bin/bash
cd /root/organism
rm -rf outputs/organism-v16
export PYTHONUNBUFFERED=1
python3 -u -m organism \
  --episodes 500 \
  --run-name organism-v16 \
  --seed 42 \
  --edge-curriculum \
  --edge-penalty 0.01 \
  --food-approach 0.03 \
  --learning-rate 0.0015 \
  --no-sleep
