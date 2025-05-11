import logging
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env

from envs.HollowGym import HollowGym, create_env
from utils.logger import LoggingCallback
from utils.websockets.servers import HollowGymServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler(sys.stdout)],
)
model_logger = configure(
    "logs/",
    ["stdout", "tensorboard"]
)
main_logger = logging.getLogger(__name__)


def main():
    n_games = 1000
    n_epochs = 400

    env = create_env("",  4649, 4, 2,"Hornet Boss 1", "GG_Hornet_1")
    check_env(env, warn=True, skip_render_check=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./checkpoints/",
        name_prefix="ppo_hornet_v2",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    logging_callback = LoggingCallback(verbose=1, log_every_steps=500)
    env_callback = CallbackList([checkpoint_callback, logging_callback])


    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=8e-5,
        batch_size=32,
        train_freq=(4, "step"),
        tensorboard_log="logs/",
    )
    model.set_logger(model_logger)
    model.learn(
        total_timesteps=n_epochs * n_games,
        callback=env_callback,
        tb_log_name="Main training loop"
    )
    model.save("checkpoints/ppo_hornet_v2")

if (__name__ == "__main__"):
    main()
