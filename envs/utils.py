import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from utils.websockets.servers import HollowGymServer

import envs

def create_env(
        frame_skip: int, game_speed: float, boss_name: str, scene_name: str, target_framerate : int = 400,
        disable_rendering: bool = False, server_ip: str = "127.0.0.1", server_port: int = 4649, n_envs: int = 1
) -> gym.Env:
    hollow_server = HollowGymServer(
        server_ip=server_ip, server_port=server_port, n_clients=n_envs,
        client_settings={
            "BossSceneName": scene_name, "BossName": boss_name,
            "FrameSkip": frame_skip, "GameSpeed": game_speed,
            "DisableRendering": disable_rendering, "FrameCap": target_framerate,
        }
    )
    hollow_server.ready.wait()
    env = make_vec_env("envs/HollowKnight-v0", env_kwargs={"settings": hollow_server.client_settings}, n_envs=n_envs)
    for i in range(n_envs):
        env.env_method("set_client", hollow_server.clients[i], indices=[i])

    return env