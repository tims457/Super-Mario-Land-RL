#!/bin/bash


cd ~/dev/Super-Mario-Land-RL
source .venv/bin/activate
cd ~/dev/PufferLib
ls -lt experiments | head
# Eval a local checkpoint:
python demo.py --env super_mario_land --mode eval --eval-model-path $1 --vec serial --render-mode human

# Useful for finding your latest checkpoint:
