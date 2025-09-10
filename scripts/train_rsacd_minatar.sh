#!/usr/bin/env bash
set -euo pipefail

TORSOS=(gru ffm slstm mlstm xlstm gtrxl shm)
ENVS=minatar_breakout,minatar_asterix,minatar_spaceinvaders,minatar_freeway

for t in "${TORSOS[@]}"; do
  python main.py -m \
    algorithm=rsacd \
    algorithm/torso@algorithm.{actor,critic}.torso="$t" \
    environment="${ENVS}" \
    seed=0 \
    num_seeds=5 \
    logger=[wandb,file] \
    hydra/launcher=julia2 &
done

wait

