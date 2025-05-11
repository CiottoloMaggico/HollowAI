import asyncio
import logging
import sys

import gymnasium as gym
import envs
import matplotlib.pyplot as plt
import numpy as np

from core.Agent import Agent
from envs.HollowGym import HollowGym
from utils.websockets.servers import HollowGymServer


# Create log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/debug.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def plot_learning_curve(x, scores, figure_file):
    running_avg = [np.mean(scores[max(0, i-100):(i+1)]) for i in range(len(scores))]

    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

async def main():
    N = 10  # learning frequency
    batch_size = 5
    n_games = 5000
    n_epochs = 200
    alpha = 2.5e-4  # learning rate

    socket_server = HollowGymServer("", 4649)
    await socket_server.mod_client_ready.wait()
    env = HollowGym(socket_server = socket_server)
    agent = Agent(
        env=env,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs
    )

    try:
        agent.load_models()
    except FileNotFoundError:
        logging.warning("No saved models found, proceeding anyway from scratch")

    best_score = -450
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    logger.info("Initialization done, starting training")
    for i in range(n_games):
        logging.info(f"Starting game {i}")
        observation, _ = await env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.choose_action(observation)
            logger.info(f"action chosen: {action}")

            observation_, reward, done, _, _ = await env.step(action)
            n_steps += 1
            score += reward
            logger.info(f"new obs:\n{observation_}")

            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()


        logging.info("Test")

        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, "ppo_learning_curve.png")

        logging.info(
            f"episode: {i}, score: {score}, avg_score: {avg_score}, time_steps: {n_steps}, learning_steps: {learn_iters}"
        )

if (__name__ == "__main__"):
    asyncio.run(main())
