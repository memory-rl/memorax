python main.py -m \
  algorithm=drqn \
  algorithm/torso=gru,slstm,mlstm,gtrxl,shm,ffm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256 \
  num_seeds=5  \
  logger=file \
  hydra/launcher=julia2 &

