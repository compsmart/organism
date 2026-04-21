#!/bin/bash
cd /root/organism
rm -rf outputs/organism-v15
export PYTHONUNBUFFERED=1
python3 -u -m organism --episodes 300 --run-name organism-v15 --seed 42 --no-sleep
