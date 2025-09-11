# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=gru \
#   algorithm/torso@algorithm.critic.torso=gru \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#
# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=ffm \
#   algorithm/torso@algorithm.critic.torso=ffm \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#
# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=slstm \
#   algorithm/torso@algorithm.critic.torso=slstm \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#
# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=mlstm \
#   algorithm/torso@algorithm.critic.torso=mlstm \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#
# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=xlstm \
#   algorithm/torso@algorithm.critic.torso=xlstm \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#
# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=gtrxl \
#   algorithm/torso@algorithm.critic.torso=gtrxl \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#
# python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=shm \
#   algorithm/torso@algorithm.critic.torso=shm \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=16,32,64,128,256 \
# num_seeds=5 \
# logger=[wandb,file] \
# hydra/launcher=julia2 &
#

#!/usr/bin/env bash
set -euo pipefail

TORSOS=(optimized_lstm gru ffm slstm mlstm xlstm gtrxl shm)
LENGTHS=(16 32 64 128 256 512 1024)

for t in "${TORSOS[@]}"; do
  for l in "${LENGTHS[@]}"; do
    python main.py -m \
      algorithm=rppo \
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


