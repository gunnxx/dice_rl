#!/bin/bash
source /home/twguntara/.bashrc

export D4RL_DATASET_DIR="/home/twguntara/.d4rl"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-440

cd /ext2/twguntara/dice_rl
/home/twguntara/miniconda3/envs/dice-rl/bin/python scripts/run_neural_dice_condor.py --cid $1 --pid $2