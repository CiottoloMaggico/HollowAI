import datetime
import logging
import os.path
import sys
import yaml

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from envs.utils import create_env
from utils.logger import LoggingCallback, EvaluationLogger

# Create log
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def load_config():
    logger.info("Loading config")
    with open("config.yaml") as conf:
        conf = yaml.load(conf, Loader=yaml.FullLoader)

    logger.info("Config loaded, validating...")
    assert(not conf["run"]["load_existing"] or conf["agent"]["load_model"])
    assert(conf["run"]["action"] == "training" or conf["run"]["action"] == "evaluation")
    logger.info("Config successfully loaded and validated")

    return conf["gym"], conf["agent"], conf["callbacks"], conf["training"], conf["evaluation"], conf["run"]

def main():
    gym, agent, callbacks, train, eval, run = load_config()
    model_name = f"{gym['boss_name']}_{run['action']}"

    env = create_env(**gym)
    if run["load_existing"]:
        logger.info("Loading existing model")
        model = DQN.load(f"./checkpoints/{agent['load_model']}", env=env, **agent["hyperparameters"])
        if agent["load_replay_buffer"]: model.load_replay_buffer(path=f"./checkpoints/{agent['load_replay_buffer']}")
        logger.info(f"Successfully loaded existing model")
    else:
        logger.info("Creating new model")
        model = DQN("MlpPolicy", env=env, verbose=1, tensorboard_log="./logs/", **agent["hyperparameters"])
        logger.info(f"Successfully created new model")

    if run["action"] == "training":
        logger.info("Creating training callbacks")
        checkpoint_callback = CheckpointCallback(save_path="./checkpoints/", name_prefix=model_name, **callbacks["checkpoint_callback"])
        logging_callback = LoggingCallback(verbose=1)
        env_callback = CallbackList([checkpoint_callback, logging_callback])
        logger.info("Callbacks ready, starting training")

        model.learn(callback=env_callback, tb_log_name=model_name, **train)

        logger.info("Model training completed, saving model...")
        model.save(f"./checkpoints/{model_name}")
        logger.info("Model saved")
    else:
        eval_callback = EvaluationLogger(f"./logs/{model_name}", gym["n_envs"])
        evaluate_policy(model, env, callback=eval_callback, deterministic=True, **eval)

if (__name__ == "__main__"):
    main()
