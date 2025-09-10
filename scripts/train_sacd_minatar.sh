python main.py -m \
  algorithm=sacd \
  environment=minatar_asterix,minatar_breakout,minatar_spaceinvaders,minatar_freeway \
  seed=0 \
  num_seeds=5 \
  logger=[wandb,file] \
  hydra/launcher=julia2 &

