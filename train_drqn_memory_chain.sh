python main.py -m \
  algorithm=drqn \
  algorithm/torso=optimized_lstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=gru \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=slstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=mlstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=xlstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=ffm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=shm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=gtrxl \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=drqn \
  algorithm/torso=gpt2 \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &
