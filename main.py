import logging
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from envs.HollowGym import HollowGym, create_env
from utils.logger import LoggingCallback
from utils.websockets.servers import HollowGymServer


# Create log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler(sys.stdout)],
)
model_logger = configure(
    "logs/",
    ["tensorboard"]
)
main_logger = logging.getLogger(__name__)


def main():
    total_time_steps = 1_000_000
    env = create_env(4649, 4, 2,"Hornet Boss 1", "GG_Hornet_1", 87)
    logger.info("Vectorized environment ready")

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="./checkpoints/",
        name_prefix="ppo_hornet_v2",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    logging_callback = LoggingCallback(verbose=1, log_every_steps=500)
    env_callback = CallbackList([checkpoint_callback, logging_callback])
    logger.info("Environment callbacks ready")

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_starts=5000,   # to not reinforce bad guesses due to initial exploration (+ let buffer fill up)
        learning_rate=7e-5,     # how big each update to the Q-network weights is during training.
        gamma=0.95,             # discount factor: how much an agent prioritizes future rewards over immediate ones
        tau=1,                  # soft update coeff: how fast the target network moves toward the online network
        buffer_size=100_000,
        batch_size=64,
        train_freq=(4, "step"),
        gradient_steps=1,       # how many gradient updates per step
        exploration_initial_eps=1.0,    # start exploration rate
        exploration_final_eps=0.3,      # end exploration rate
        exploration_fraction=0.7,       # expl. rate will linearly decrease from start to end in (exploration_fraction * total_timesteps) steps
        tensorboard_log="logs/",
    )
    model.set_logger(model_logger)

    logger.info("Model setup completed, starting training...")
    model.learn(
        total_timesteps=total_time_steps,
        callback=env_callback,
        tb_log_name="Main training loop"
    )
    logger.info("Model training completed, saving model...")
    model.save("checkpoints/ppo_hornet_v2")
    logger.info("Model saved")

if (__name__ == "__main__"):
    main()
