uv run python main.py -m \
  algorithm=drqn \
  algorithm/torso=slstm,mlstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

# uv run python main.py -m \
#   algorithm=rppo \
#   algorithm/torso@algorithm.actor.torso=optimized_lstm \
#   algorithm/torso@algorithm.critic.torso=optimized_lstm \
#   environment=bsuite_memory_chain \
#   environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
#   seed=0,1,2,3,4 \
#   logger=wandb \
#   hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=gru \
  algorithm/torso@algorithm.critic.torso=gru \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=ffm \
  algorithm/torso@algorithm.critic.torso=ffm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=slstm \
  algorithm/torso@algorithm.critic.torso=slstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rppo \
  algorithm/torso@algorithm.actor.torso=mlstm \
  algorithm/torso@algorithm.critic.torso=mlstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rsacd \
  algorithm/torso@algorithm.actor.torso=optimized_lstm \
  algorithm/torso@algorithm.critic.torso=optimized_lstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=console \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rsacd \
  algorithm/torso@algorithm.actor.torso=gru \
  algorithm/torso@algorithm.critic.torso=gru \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rsacd \
  algorithm/torso@algorithm.actor.torso=ffm \
  algorithm/torso@algorithm.critic.torso=ffm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rsacd \
  algorithm/torso@algorithm.actor.torso=slstm \
  algorithm/torso@algorithm.critic.torso=slstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1

uv run python main.py -m \
  algorithm=rsacd \
  algorithm/torso@algorithm.actor.torso=mlstm \
  algorithm/torso@algorithm.critic.torso=mlstm \
  environment=bsuite_memory_chain \
  environment.parameters.memory_length=1,2,4,8,16,32,64,128 \
  seed=0,1,2,3,4 \
  logger=wandb \
  hydra/launcher=joblib hydra.launcher.n_jobs=1
