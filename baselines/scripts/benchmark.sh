#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TORSO="${1:-mlp}"
ENV="${2:-popjym/autoencode/easy}"

uv run benchmark.py \
  algorithm=pqn \
  environment="$ENV" \
  torso="$TORSO" \
  num_seeds=1 \
  num_train_steps=16384 \
  total_timesteps=16384 \
  logger=default
