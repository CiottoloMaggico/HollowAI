import logging

import gymnasium as gym
import numpy as np

from utils.websockets.exceptions import ModClientNotConnected
from utils.websockets.servers import HollowGymServer

logger = logging.getLogger(__name__)
asyncioLogger = logging.getLogger("asyncio")


class HollowGym:
    def __init__(self, socket_server : HollowGymServer):
        self.socket_server = socket_server

        self.action_space = gym.spaces.Discrete(4**4)
        self._action_to_action_code = {
            i : np.base_repr(i, 4, 5)[-4:] for i in range(4**4)
        }

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(97,), dtype=np.float32)

    def _get_observation(self, message):
        obs = message["Data"]["Observation"]
        return np.array(
            [
                obs["PlayerHpPerc"],
                obs["PlayerMpPerc"],
                obs["PlayerReserveMpPerc"],
                *obs["PlayerPos"],
                obs["PlayerFacingRight"],
                obs["BossHpPerc"],
                *obs["BossPos"],
                obs["BossFacingRight"],
                obs["PlayerBossDistance"],
                *obs["BossFsmStateOneHot"],
            ], dtype=np.float32,
        )

    async def reset(self, seed=None, options=None):
        try:
            response = await self.socket_server.message_exchange(1)
        except ModClientNotConnected:
            await self.socket_server.mod_client_ready.wait()
            return await self.reset()

        obs = self._get_observation(response)
        info = None
        return obs, info

    async def step(self, action):
        try:
            response = await self.socket_server.message_exchange(2, self._action_to_action_code[action])
        except ModClientNotConnected:
            await self.socket_server.mod_client_ready.wait()
            return await self.reset()

        obs = self._get_observation(response)
        reward = response["Data"]["MetaData"]["Reward"]
        terminated = response["Data"]["MetaData"]["Terminated"]
        truncated = False
        info = None

        return obs, reward, terminated, truncated, info
