import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):

    def __init__(self, verbose: int = 0, log_every_steps: int = 1000):
        super().__init__(verbose=verbose)
        self.log_every_steps = log_every_steps

    def _on_training_start(self) -> None:
        self.agent_health = [[]] * self.training_env.num_envs
        self.boss_health = [[]] * self.training_env.num_envs
        self.wins_n = 0
        self.episodes_n = 0

    def _on_step(self) -> bool:
        new_obs = self.locals["new_obs"]
        dones = self.locals["dones"]

        for i in range(0, self.training_env.num_envs):
            obs_n, done_n = new_obs[i], dones[i]
            self.agent_health[0] += [obs_n[0]]
            self.boss_health[0] += [obs_n[6]]
            # TODO: add win in the "info" received from the mod client
            self.wins_n += 1 if obs_n[6] == 0 and obs_n[0] > 0 and done_n else 0
            self.episodes_n += 1 if done_n else 0

            if (done_n):
                self.logger.record("custom/avg_agent_health", np.mean(self.agent_health))
                self.logger.record("custom/avg_boss_health", np.mean(self.boss_health))

                self.agent_health[i].clear()
                self.boss_health[i].clear()


        if self.num_timesteps % self.log_every_steps == 0:
            self.logger.record("custom/win_rate", self.wins_n/self.episodes_n)

        return True