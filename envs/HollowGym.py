import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from utils.websockets.exceptions import ModClientNotConnected
from utils.websockets.servers import HollowGymServer


def create_env(server_port: int, frame_skip: int, game_speed: float, boss_name: str, scene_name: str,
               observation_size : int, server_ip: str = "127.0.0.1", n_envs: int = 1):
    hollow_server = HollowGymServer(
        server_ip=server_ip, server_port=server_port, n_clients=n_envs,
        client_settings={
            "BossSceneName": scene_name, "BossName": boss_name,
            "FrameSkip": frame_skip, "GameSpeed": game_speed,
        }
    )
    hollow_server.ready.wait()
    env = make_vec_env(HollowGym, env_kwargs={"observation_size": observation_size})
    for i in range(n_envs):
        env.env_method("set_client", hollow_server.clients[i], indices=i)

    return env

class HollowGym(gym.Env):
    def __init__(self, observation_size: int):
        self.client = None
        self.observation_size = observation_size
        self._action_to_action_code = {
            i: np.base_repr(i, 4, 5)[-4:] for i in range(4 ** 4)
        }

        self.action_space = gym.spaces.Discrete(4**4)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.observation_size,), dtype=np.float32)

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
            response = self.client.message_exchange(1)
        except ModClientNotConnected:
            self.client.mod_client_ready.wait()
            return self.reset()

        obs = self._get_observation(response)
        info = response["Data"]["Info"]
        return obs, info

    def step(self, action):
        try:
            response = self.client.message_exchange(2, self._action_to_action_code[action])
        except ModClientNotConnected:
            self.client.mod_client_ready.wait()
            return self.reset()

        obs = self._get_observation(response)
        info = response["Data"]["Info"]
        reward = info["Reward"]
        terminated = info["Terminated"]
        truncated = False

        return obs, reward, terminated, truncated, info

    def set_client(self, client):
        self.client = client