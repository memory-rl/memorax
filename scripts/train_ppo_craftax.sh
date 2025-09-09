python main.py -m \
  algorithm=ppo \
  environment=craftax_symbolic \
  num_seeds=5 \
logger=[wandb,file] \
hydra/launcher=julia2 &

