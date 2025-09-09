python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gru \
  algorithm/torso@algorithm.critic.torso=gru \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=ffm \
  algorithm/torso@algorithm.critic.torso=ffm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=slstm \
  algorithm/torso@algorithm.critic.torso=slstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=mlstm \
  algorithm/torso@algorithm.critic.torso=mlstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=xlstm \
  algorithm/torso@algorithm.critic.torso=xlstm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gtrxl \
  algorithm/torso@algorithm.critic.torso=gtrxl \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=shm \
  algorithm/torso@algorithm.critic.torso=shm \
  environment=bsuite_umbrella_chain \
  environment.parameters.chain_length=16,32,64,128 \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

