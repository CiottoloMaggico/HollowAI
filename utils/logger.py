from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

class TensorboardLogger(CheckpointCallback):

    def __init__(self, save_freq, save_path, name_prefix="model", verbose=0):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)
        self.agent_health = []
        self.boss_health = []
        self.wins = 0

    def _on_step(self) -> bool:

        super()._on_step()
        obs = self.locals.get("new_obs", [])
        dones = self.locals.get("dones", [])

        self.agent_health.append(obs[0, 0])
        self.boss_health.append(obs[0, 6])
        if(dones[0]): self.wins+=1

        return True # False to stop the model : could be used for Early Stopping

    def _on_rollout_end(self) -> None:
        # Called after each rollout. Save metrics to file or plot
        self.logger.record("custom/avg_agent_health", np.mean(self.agent_health)) # should increase over time
        self.logger.record("custom/avg_boss_health", np.mean(self.boss_health))   # should decrease over time
        self.logger.record("custom/win_rate", self.wins/self.save_freq) # HOPE it increases

        self.boss_health.clear()
        self.agent_health.clear()
        self.wins = 0


