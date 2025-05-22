import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean
from torch.utils.tensorboard import SummaryWriter


class LoggingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.boss_health = None
        self.episode_rewards = None
        self.wins_n = None
        self.episodes_n = None
        self.tb_formatter = None

    def _on_training_start(self) -> None:
        self.boss_health = [[] for _ in range(self.training_env.num_envs)]
        self.episode_rewards = [0] * self.training_env.num_envs
        self.wins_n = 0
        self.episodes_n = 0
        self.tb_formatter = next(formatter for formatter in self.logger.output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        new_observations = self.locals["new_obs"]
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i in range(0, self.training_env.num_envs):
            reward, info, done = rewards[i], infos[i], dones[i]
            self._log_step(i, new_observations[i], rewards[i], infos[i], dones[i])

        return True

    def _log_step(self, i, obs, reward, info, done):
        self.boss_health[i] += [obs[27]]
        self.episode_rewards[i] += reward
        self.wins_n += 1 if info["Win"] else 0
        self.episodes_n += 1 if done else 0

        if not done: return
        self.tb_formatter.writer.add_scalar("episode/min_boss_health", np.min(self.boss_health[i]), self.episodes_n)
        self.tb_formatter.writer.add_scalar("episode/reward", self.episode_rewards[i], self.episodes_n)
        self.tb_formatter.writer.add_scalar("episode/win_rate", self.wins_n / self.episodes_n, self.episodes_n)
        self.tb_formatter.writer.add_scalar("episode/wins", self.wins_n, self.episodes_n)
        self.tb_formatter.writer.flush()

        self.episode_rewards[i] = 0
        self.boss_health[i].clear()

        return

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/episodes", self.episodes_n)


class EvaluationLogger:
    def __init__(self, log_dir : str, n_envs : int = 1):
        self.log_dir = log_dir
        self.n_envs = n_envs
        self.boss_health = [[] for _ in range(self.n_envs)]
        self.episode_rewards = [0] * self.n_envs
        self.episode_steps = [0] * self.n_envs
        self.reward_history = []
        self.steps_history = []
        self.wins_n = 0
        self.episodes_n = 0
        self.writer = SummaryWriter(self.log_dir)

    def __call__(self, locals, globals):
        i = locals["i"]
        obs, reward, info, done = locals["new_observations"][i], locals["reward"], locals["info"], locals["done"]
        self.boss_health[i] += [obs[27]]
        self.episode_rewards[i] += reward
        self.episode_steps[i] += 1
        self.wins_n += 1 if info["Win"] else 0
        self.episodes_n += 1 if done else 0
        self.reward_history += [self.episode_rewards[i]] if done else []
        self.steps_history += [self.episode_steps[i]] if done else []

        if not done: return
        self.writer.add_scalar("eval/min_boss_health", np.min(self.boss_health[i]), self.episodes_n)
        self.writer.add_scalar("eval/reward", self.episode_rewards[i], self.episodes_n)
        self.writer.add_scalar("eval/win_rate", self.wins_n / self.episodes_n, self.episodes_n)
        self.writer.add_scalar("eval/wins", self.wins_n, self.episodes_n)
        self.writer.add_scalar("eval/ep_rew_mean", safe_mean(self.reward_history), self.episodes_n)
        self.writer.add_scalar("eval/ep_len_mean", safe_mean(self.steps_history), self.episodes_n)
        self.writer.flush()

        self.episode_rewards[i] = 0
        self.episode_steps[i] = 0
        self.boss_health[i].clear()



