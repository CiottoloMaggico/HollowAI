from gymnasium.envs.registration import register

register(
    id="envs/HollowKnight-v0",
    entry_point="envs.hollow_gym:HollowKnightEnv",
)