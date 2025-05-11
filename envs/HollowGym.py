import gymnasium as gym
import numpy as np

from utils.websockets.exceptions import ModClientNotConnected
from utils.websockets.servers import HollowGymServer

def create_env(server_ip: str, server_port: int, frame_skip : int, game_speed: float, boss_name : str, scene_name : str):
    socket_server = HollowGymServer("", 4649, frame_skip, game_speed, boss_name, scene_name)
    socket_server.start()

    socket_server.mod_client_ready.wait()
    env = HollowGym(socket_server = socket_server)
    return env

class HollowGym(gym.Env):
    def __init__(self, socket_server : HollowGymServer):
        self.socket_server = socket_server

        self.action_space = gym.spaces.Discrete(4**4)
        self._action_to_action_code = {
            i : np.base_repr(i, 4, 5)[-4:] for i in range(4**4)
        }
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(89,), dtype=np.float32)

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

    def reset(self, seed=None, options=None):
        try:
            response = self.socket_server.message_exchange(1)
        except ModClientNotConnected:
            self.socket_server.mod_client_ready.wait()
            return self.reset()

        obs = self._get_observation(response)
        info = response["Data"]["Info"]
        return obs, info

    def step(self, action):
        try:
            response = self.socket_server.message_exchange(2, self._action_to_action_code[action])
        except ModClientNotConnected:
            self.socket_server.mod_client_ready.wait()
            return self.reset()

        obs = self._get_observation(response)
        info = response["Data"]["Info"]
        reward = info["Reward"]
        terminated = info["Terminated"]
        truncated = False

        return obs, reward, terminated, truncated, info
