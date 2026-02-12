#!/bin/bash

TORSOS="gru lstm mingru s5 slstm self_attention gtrxl"

ENVS="popjym/autoencode/easy"
ENVS="${ENVS},popjym/concentration/easy"
ENVS="${ENVS},popjym/count_recall/easy"
ENVS="${ENVS},popjym/repeat_first/easy"
ENVS="${ENVS},popjym/repeat_previous/easy"

for TORSO in $TORSOS; do
  sbatch --export=ALL -J "popjym-${TORSO}" --gres=gpu:1 --mem-per-cpu 2G -t 06:00:00 -p stud -c 4 \
    --wrap="uv run main.py -m \
              algorithm=ppo \
              environment=${ENVS} \
              torso=${TORSO} \
              seed=0 \
              logger=[dashboard,wandb]"
done
