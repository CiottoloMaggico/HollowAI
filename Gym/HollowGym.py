import json
import logging

import gymnasium as gym
import numpy as np

from .WebSocketGym import WebSocketGym

logger = logging.getLogger(__name__)
asyncioLogger = logging.getLogger("asyncio")

class HollowGym(WebSocketGym):
    def __init__(self, server_ip: str, server_port: int):
        super().__init__(server_ip, server_port)

        self.action_space_dim = 4**4
        self.action_space = gym.spaces.Discrete(self.action_space_dim)
        self._action_to_action_code = {
            i : np.base_repr(i, 4, 5)[-4:] for i in range(self.action_space_dim)
        }

        self.observation_space = gym.spaces.Dict({
            "PlayerHpPerc" : gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            "PlayerMpPerc" : gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            "PlayerReserveMpPerc" : gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            "PlayerPos" : gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "PlayerFacingRight" : gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            "BossHpPerc" : gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            "BossPos" : gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "BossFacingRight" : gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            "BossFsmStateOneHot" : gym.spaces.Box(low=0.0, high=1.0, shape=(85,),  dtype=np.float32),
        })

    @staticmethod
    def preprocess_observation(obs: dict) -> np.ndarray:
        if obs["BossFsmStateOneHot"] is None:
            obs["BossFsmStateOneHot"] = [0] * 85

        flat = np.array([
            obs["PlayerHpPerc"],
            obs["PlayerMpPerc"],
            obs["PlayerReserveMpPerc"],
            *obs["PlayerPos"],
            obs["PlayerFacingRight"],
            obs["BossHpPerc"],
            *obs["BossPos"],
            obs["BossFacingRight"],
            *obs["BossFsmStateOneHot"],
        ], dtype=np.float32)
        return flat


    async def reset(self, seed=None, options=None):
        res = await self._message_exchange(1)
        if res is None or res["Cmd"] != 1: return None

        obs = HollowGym.preprocess_observation(res["Data"]["Observation"])
        info = None
        return obs, info

    async def step(self, action):
        res = await self._message_exchange(2, self._action_to_action_code[action])
        if res is None or res["Cmd"] != 2: return None

        obs = HollowGym.preprocess_observation(res["Data"]["Observation"])
        reward = res["Data"]["MetaData"]["Reward"]
        terminated = res["Data"]["MetaData"]["Terminated"]
        truncated = False
        info = None

        return obs, reward, terminated, truncated, info
