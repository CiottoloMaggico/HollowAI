import logging
import datetime

import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class LoggingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_training_start(self) -> None:
        self.agent_health = [[] for _ in range(self.training_env.num_envs)]
        self.boss_health = [[] for _ in range(self.training_env.num_envs)]
        self.episode_rewards = [0] * self.training_env.num_envs
        self.wins_n = 0
        self.episodes_n = 0
        self.tb_formatter = next(formatter for formatter in self.logger.output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        new_obs = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i in range(0, self.training_env.num_envs):
            obs_n, reward_n, info_n, done_n = new_obs[i], rewards[i], infos[i], dones[i]
            self.agent_health[0] += [obs_n[0]]
            self.boss_health[0] += [obs_n[27]]
            self.wins_n += 1 if info_n["Win"] else 0
            self.episodes_n += 1 if done_n else 0
            self.episode_rewards[i] += reward_n

            if not done_n: continue
            self.tb_formatter.writer.add_scalar("episode/min_boss_health", np.min(self.boss_health), self.episodes_n)
            self.tb_formatter.writer.add_scalar("episode/reward", self.episode_rewards[i], self.episodes_n)
            self.tb_formatter.writer.add_scalar("episode/win_rate", self.wins_n/self.episodes_n, self.episodes_n)
            self.tb_formatter.writer.add_scalar("episode/wins", self.wins_n, self.episodes_n)
            self.tb_formatter.writer.flush()

            self.episode_rewards[i] = 0
            self.agent_health[i].clear()
            self.boss_health[i].clear()

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/episodes", self.episodes_n)