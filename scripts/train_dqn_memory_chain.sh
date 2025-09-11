python main.py -m \
  algorithm=dqn \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
  seed=0 \
  num_seeds=5 \
  logger=[wandb,file] \
  hydra/launcher=julia2 &

