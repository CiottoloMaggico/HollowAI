import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from envs.HollowGym import HollowGym
from utils.websockets.servers import HollowGymServer
from utils.logger import TensorboardLogger

new_logger = configure("logs/", ["stdout", "log", "tensorboard"])
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    n_games = 1000
    n_epochs = 400

    socket_server = HollowGymServer("", 4649)
    socket_server.start()
    socket_server.mod_client_ready.wait()

    env = HollowGym(socket_server = socket_server)
    #check_env(env, warn=True, skip_render_check=True)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=8e-5,
        batch_size=32,
        train_freq=(4, "step"),
        tensorboard_log="logs/",
    )

    model.set_logger(new_logger)
    callback = TensorboardLogger(save_freq=10e3, save_path="logs/saves/", name_prefix="dqn_model")

    model.learn(total_timesteps=n_epochs * n_games, callback=callback)
    model.save("ppo_hornet_v2")

if (__name__ == "__main__"):
    main()
