#!/usr/bin/env bash
set -euo pipefail

TORSOS="mlp,gru,lstm,s5,mamba,lru,mingru,mlstm,slstm,ffm,shm,linear_attention,self_attention"
ENVS="popjym/autoencode/easy,popjym/autoencode/medium,popjym/autoencode/hard,popjym/count_recall/easy,popjym/count_recall/medium,popjym/count_recall/hard,popjym/repeat_first/easy,popjym/repeat_first/medium,popjym/repeat_first/hard,popjym/repeat_previous/easy,popjym/repeat_previous/medium,popjym/repeat_previous/hard,popjym/concentration/easy,popjym/concentration/medium,popjym/concentration/hard"

uv run main.py -m \
  algorithm=pqn \
  environment=$ENVS \
  torso=$TORSOS \
  num_seeds=1 \
  logger=[dashboard,wandb]
