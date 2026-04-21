#!/bin/bash
cd /root/organism
rm -rf outputs/organism-v14
export PYTHONUNBUFFERED=1
python3 -u -m organism --episodes 300 --run-name organism-v14 --seed 42
