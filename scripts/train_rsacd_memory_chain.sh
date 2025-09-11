#!/usr/bin/env bash
set -euo pipefail

TORSOS=(optimized_lstm gru ffm slstm mlstm xlstm gtrxl shm)
LENGTHS=(16 32 64 128 256 512 1024)

for t in "${TORSOS[@]}"; do
  for l in "${LENGTHS[@]}"; do
    python main.py -m \
      algorithm=rsacd \
      algorithm/torso@algorithm.{actor,critic}.torso="$t" \
      environment=bsuite_memory_chain \
      environment.parameters.memory_length="$l" \
      seed=0 \
      num_seeds=5 \
      logger=[wandb,file] \
      hydra/launcher=julia2 &
    done
done

wait


