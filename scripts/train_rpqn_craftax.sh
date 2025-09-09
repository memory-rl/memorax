python main.py -m \
  algorithm=rpqn \
  algorithm/torso=gru \
  algorithm/torso=gru \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=slstm \
  algorithm/torso=slstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=mlstm \
  algorithm/torso=mlstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=xlstm \
  algorithm/torso=xlstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=gtrxl \
  algorithm/torso=gtrxl \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rpqn \
  algorithm/torso=ffm \
  algorithm/torso=ffm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=julia2 &
