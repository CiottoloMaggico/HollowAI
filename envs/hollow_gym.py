import gymnasium as gym
import numpy as np

from utils.websockets.exceptions import ModClientNotConnected
from utils.websockets.servers import HollowGymServer

class HollowKnightEnv(gym.Env):
    def __init__(self, settings):
        self._client = None
        self._settings = settings
        self._action_to_action_code = {
            i: np.base_repr(i, 4, 5)[-4:] for i in range(4 ** 4)
        }

        self.action_space = gym.spaces.Discrete(4**4)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._settings["ObservationSize"],), dtype=np.float32)

    def close(self):
        if self.socket_server.mod_client:
            self.socket_server.server_thread.stop()

    def _get_observation(self, message):
        obs = message["Data"]["Observation"]
        return np.array(
            [
                obs["PlayerHpPerc"],
                obs["PlayerMpPerc"],
                obs["PlayerReserveMpPerc"],
                *obs["PlayerPosition"],
                *obs["PlayerVelocity"],
                *obs["PlayerState"],
                obs["BossHpPerc"],
                *obs["BossPosition"],
                *obs["BossVelocity"],
                obs["BossFacingRight"],
                obs["PlayerBossDistance"],
                obs["SceneCenterDistance"],
                *obs["BossStateOneHot"],
            ], dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        try:
            response = self._client.message_exchange(1)
        except ModClientNotConnected:
            self._client.mod_client_ready.wait()
            return self.reset()

        obs = self._get_observation(response)
        info = response["Data"]["Info"]
        return obs, info

    def step(self, action):
        try:
            response = self._client.message_exchange(2, self._action_to_action_code[action])
        except ModClientNotConnected:
            self._client.mod_client_ready.wait()
            return self.reset()

        obs = self._get_observation(response)
        info = response["Data"]["Info"]
        reward = info["Reward"]
        terminated = info["Terminated"]
        truncated = False

        return obs, reward, terminated, truncated, info

    def set_client(self, client):
        self._client = client