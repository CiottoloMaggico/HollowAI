import asyncio
from Gym.HollowGym import HollowGym
from Core.Agent import Agent
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = [np.mean(scores[max(0, i-100):(i+1)]) for i in range(len(scores))]

    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

async def main():
    env = HollowGym(server_ip="", server_port="4649")

    N = 5  # learning frequency
    batch_size = 5
    n_games = 1000
    n_epochs = 4
    alpha = 0.0003  # learning rate

    agent = Agent(n_actions=255, input_dims=95,
                  batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

    best_score = 0
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    await asyncio.sleep(20)
    print("Starting training...")
    for i in range(n_games):
        observation, _ = await env.reset()
        done = False
        score = 0

        while not done:
            print("choosing action...")
            action, prob, val = agent.choose_action(observation)
            print(f"action choosed: {action} {prob} {val}")

            # (STEP)
            # Send action to WebSocket
            # Receive observation, reward, done from WebSocket
            print("move one step forward")
            observation_, reward, done, _, _ = await env.step(action)
            n_steps += 1
            score += reward
            print("new obs: ", observation_)

            # Buffer the experience
            print("buffer the experience")
            agent.remember(observation, action, prob, val, reward, done)
            observation = observation_
            print("experience buffered")

            # Every N steps, learn from buffered experience
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, "ppo_learning_curve.png")

if (__name__ == "__main__"):
    asyncio.run(main())
