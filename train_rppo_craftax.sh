python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=optimized_lstm \
  algorithm/torso@algorithm.critic.torso=optimized_lstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gru \
  algorithm/torso@algorithm.critic.torso=gru \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &


python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=slstm \
  algorithm/torso@algorithm.critic.torso=slstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=mlstm \
  algorithm/torso@algorithm.critic.torso=mlstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=xlstm \
  algorithm/torso@algorithm.critic.torso=xlstm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gtrxl \
  algorithm/torso@algorithm.critic.torso=gtrxl \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=shm \
  algorithm/torso@algorithm.critic.torso=shm \
  environment=craftax_symbolic \
seed=0,1,2,3,4 \
logger=[wandb,file] \
hydra/launcher=lichtenberg &

