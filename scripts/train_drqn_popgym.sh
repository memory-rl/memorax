python main.py -m \
  algorithm=drqn \
  algorithm/torso=optimized_lstm,gru,slstm,mlstm,xlstm,ffm,shm,gtrxl,gpt2 \
  environment=popgym_autoencode_easy,popgym_autoencode_hard,popgym_battleship_easy,popgym_battleship_hard,popgym_concentration_easy,popgym_concentration_hard,popgym_count_recall_easy,popgym_count_recall_hard,popgym_higher_lower_easy,popgym_higher_lower_hard,popgym_repeat_first_easy,popgym_repeat_first_hard,popgym_stateless_cartpole_easy,popgym_stateless_cartpole_hard \
seed=0,1,2,3,4 \
logger=wandb \
hydra/launcher=julia2 &

