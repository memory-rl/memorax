python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=optimized_lstm \
  algorithm/torso@algorithm.critic.torso=optimized_lstm \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gru \
  algorithm/torso@algorithm.critic.torso=gru \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=ffm \
  algorithm/torso@algorithm.critic.torso=ffm \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=slstm \
  algorithm/torso@algorithm.critic.torso=slstm \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=mlstm \
  algorithm/torso@algorithm.critic.torso=mlstm \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=xlstm \
  algorithm/torso@algorithm.critic.torso=xlstm \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gtrxl \
  algorithm/torso@algorithm.critic.torso=gtrxl \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=shm \
  algorithm/torso@algorithm.critic.torso=shm \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

