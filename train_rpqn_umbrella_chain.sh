python main.py -m \
  algorithm=rpqn \
  algorithm/torso=optimized_lstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=gru \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=slstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=mlstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=xlstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=ffm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=shm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=gtrxl \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=gpt2 \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128,256,512,1024 \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=lichtenberg &

