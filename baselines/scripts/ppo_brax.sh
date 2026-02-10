#!/usr/bin/env bash
set -euo pipefail

TORSOS="mlp gru linear_transformer delta_net gated_delta_net"

for TORSO in $TORSOS; do
    echo "Running experiment with torso: $TORSO"
    uv run main.py -m \
      torso=$TORSO \
      embeddings=oa \
      environment=brax/ant \
      ++environment.kwargs.mode=F \
      seed=1 \
      logger=[dashboard,wandb]
done
