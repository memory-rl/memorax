python main.py -m \
  algorithm=pqn \
  environment=minatar_asterix,minatar_breakout,minatar_spaceinvaders,minatar_freeway,minatar_seaquest \
  seed=0 \
  num_seeds=5 \
  logger=[wandb,file] \
  hydra/launcher=lichtenberg &

