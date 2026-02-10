#!/usr/bin/env bash
set -euo pipefail

uv run main.py -m \
  algorithm=pqn \
  environment=popgym_arcade/breakout_easy,popgym_arcade/autoencode_easy,popgym_arcade/battleship_easy,popgym_arcade/count_recall_easy,popgym_arcade/minesweeper_easy,popgym_arcade/navigator_easy,popgym_arcade/noisy_cartpole_easy,popgym_arcade/skittles_easy,popgym_arcade/tetris_easy \
  torso=mlp \
  num_seeds=1 \
  logger=wandb
