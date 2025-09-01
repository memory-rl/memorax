python main.py -m \
  algorithm=rppo \
  algorithm/torso=optimized_lstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso=gru \
  algorithm/torso=gru \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &


python main.py -m \
  algorithm=rppo \
  algorithm/torso=slstm \
  algorithm/torso=slstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso=mlstm \
  algorithm/torso=mlstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso=xlstm \
  algorithm/torso=xlstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso=gtrxl \
  algorithm/torso=gtrxl \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso=shm \
  algorithm/torso=shm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

