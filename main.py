import datetime
import logging
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from envs.utils import create_env
from utils.logger import LoggingCallback

# Create log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler(sys.stdout)],
)

main_logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

def main():
    # Edit training parameters here
    TOTAL_TIMESTEPS = 2_000_000
    EVALUATION_EPISODES = 10
    TARGET_FRAMERATE = 100
    DISABLE_RENDERING = True
    FRAME_SKIP = 3
    N_ENVS = 6
    GAME_SPEED = 2
    BOSS_NAME = "Hornet Boss 1"
    SCENE_NAME = "GG_Hornet_1"
    MODEL_TO_LOAD = None
    REPLAY_BUFFER_TO_LOAD = None
    DO_TRAINING = True
    DO_EVAL = False
    # -----------------------------

    env = create_env(FRAME_SKIP, GAME_SPEED, BOSS_NAME, SCENE_NAME, n_envs=N_ENVS, disable_rendering=DISABLE_RENDERING,
                     target_framerate=TARGET_FRAMERATE)
    model_name = f"{BOSS_NAME}_DQN_{datetime.datetime.now()}" if not MODEL_TO_LOAD else MODEL_TO_LOAD

    logger.info("Creating model callbacks")
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="./checkpoints/",
        name_prefix=model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    logging_callback = LoggingCallback(verbose=1)
    env_callback = CallbackList([checkpoint_callback, logging_callback])
    logger.info("Environment callbacks ready, setting up the model...")


    if not MODEL_TO_LOAD:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_starts=5000,  # to not reinforce bad guesses due to initial exploration (+ let buffer fill up)
            learning_rate=3e-5,  # how big each update to the Q-network weights is during training.
            gamma=0.95,  # discount factor: how much an agent prioritizes future rewards over immediate ones
            tau=1,  # soft update coeff: how fast the target network moves toward the online network
            buffer_size=100_000,
            batch_size=64,
            train_freq=(4, "step"),
            gradient_steps=1,  # how many gradient updates per step
            exploration_initial_eps=.95,  # start exploration rate
            exploration_final_eps=.05,  # end exploration rate
            exploration_fraction=0.8, # expl. rate will linearly decrease from start to end in (exploration_fraction * total_timesteps) steps
            tensorboard_log="./logs/",
        )
    else:
        model = DQN.load(f"./checkpoints/{MODEL_TO_LOAD}", env=env)
        if REPLAY_BUFFER_TO_LOAD: model.load_replay_buffer(path=f"./checkpoints/{REPLAY_BUFFER_TO_LOAD}")

    if DO_TRAINING:
        logger.info("Model ready, starting training...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=env_callback,
            tb_log_name=model_name,
            progress_bar = True,
        )
        logger.info("Model training completed, saving model...")
        model.save(f"./checkpoints/{model_name}")
        logger.info("Model saved")

    if DO_EVAL and EVALUATION_EPISODES > 0:
        evaluate_policy(
            model, env, n_eval_episodes=EVALUATION_EPISODES, callback=env_callback,
        )

if (__name__ == "__main__"):
    main()
