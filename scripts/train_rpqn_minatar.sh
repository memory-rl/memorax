python main.py -m \
  algorithm=rpqn \
  algorithm/torso=optimized_lstm,gru,slstm,mlstm,xlstm,ffm,shm,gtrxl \
  environment=minatar_asterix,minatar_breakout,minatar_spaceinvaders,minatar_freeway,minatar_seaquest \
  seed=0 \
  num_seeds=5 \
  logger=[wandb,file] \
  hydra/launcher=lichtenberg &

