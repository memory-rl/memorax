#!/usr/bin/env bash
set -euo pipefail

TORSOS="mlp,gru,lstm,s5,mamba,lru,mingru,mlstm,slstm,ffm,shm,linear_attention,self_attention"
ENVS="popjym/autoencode_easy,popjym/autoencode_medium,popjym/autoencode_hard,popjym/count_recall_easy,popjym/count_recall_medium,popjym/count_recall_hard,popjym/repeat_first_easy,popjym/repeat_first_medium,popjym/repeat_first_hard,popjym/repeat_previous_easy,popjym/repeat_previous_medium,popjym/repeat_previous_hard,popjym/concentration_easy,popjym/concentration_medium,popjym/concentration_hard"

uv run main.py -m \
  algorithm=r2d2 \
  environment=$ENVS \
  torso=$TORSOS \
  num_seeds=5 \
  logger=wandb
