
<picture>
  <source srcset="https://github.com/user-attachments/assets/84bf41bf-02b3-4b5d-86ff-d53681caa579" media="(prefers-color-scheme: light)">
  <source srcset="https://github.com/user-attachments/assets/b4d25d3a-e64b-4024-8ddb-4b667ab97b92" media="(prefers-color-scheme: dark)">
  <img src="https://github.com/user-attachments/assets/b4d25d3a-e64b-4024-8ddb-4b667ab97b92" alt="Your image description">
</picture>

This repo features a reinforcement learning agent trained to (hopefully) beat the bosses of _Hollow Knight_.

## 🧠 How it works:
HollowAi is actually an interface between: 
- **sb3 agents** 
- [**HollowGym**](https://github.com/CiottoloMaggico/HKFeeder), our custom mod that exposes a Gym-like environment for Hollow Knight

HollowGym handles the game logic: controlling the character, gathering observations, and computing rewards. Communication between HollowAI and HollowGym happens via WebSocket, using JSON-encoded messages.
### Communication protocol
The client (HollowGym) and server (HollowAI) exchange messages through a simple command system. The supported commands are:
- Exit – Clean shutdown
- Reset – Reset the environment
- Step – Perform an action
- Observation – Request current observation
- Ready – Handshake to sync server and client
- Error – Report issues

Here's a basic example of a Step command:

```json5
// server -> client
{
  "Cmd": 2, // Step
  "Data": {
    "Action": "0000", // no action
  },
}
```
```json5
// response from the client
{
  "Cmd": 2,
  "Data": {
    "Action": "0000",
    "Observation": {
      // observation after the action
    },
    "Info": {
      // some info about the environment and the current episode
    },
  }
}
```
#### The Handshake
Before any interaction can happen, the client and server perform a three-way handshake:
1. The client sends a Ready message.
2. The server responds with configuration details (from config.yaml).
3. The client confirms by returning its supported settings, including the correct observation space for the specified boss.

This ensures both components are in sync before training begins.

_Note: all the information about HollowGym protocol can be found [here](https://github.com/CiottoloMaggico/HKFeeder)_

### Stable baselines 3
Instead of reinventing the wheel (and doing it poorly), we rely on the excellent Stable Baselines3 library. It provides reliable, state-of-the-art implementations of all the RL algorithms we need.

### Getting started
---
#### Requirements: 
- A copy of **Hollow Knight**
- **HollowGym** installed and working
- This repo: **HollowAi**
#### Setup instructions
1. Create a python environment and install the dependencies.\
_If your machine have a cuda capable gpu, and you want to use it, before installing `requirements.txt` install pytorch using the command provided [here](https://pytorch.org/get-started/locally/)._
2. Edit the 'config.yaml' file to suit your setup preferencies.
3. Start **HollowAi** by running `python main.py`.
4. Launch `n_envs` instances of Hollow Knight.
5. In each instance:
   1. open a save file 
   2. once the game is loaded press `F8` to tell HollowAi that the client is ready
6. **You're all set!**

### Tensorboard Integration
---
Use `tensorboard --logdir ./logs` to open tensorboard webserver, and go in **Scalars** tab to see all the stats
