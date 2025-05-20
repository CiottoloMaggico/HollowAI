# HollowAI
A RL agent that beats Hollow Knight bosses

### Tensorboard Plots
---
Use `tensorboard --logdir ./logs` to open tensorboard webserver, and go in **Scalars** tab
- Note: it plots all the data it finds in the log folder (events.out.tfevents* files), these are not deleted after every training so you'll see everything together, consider deleting them or look at the plots with Relative or Wall format

### Reward Function
---
- **Reward Shaping**: reward shaping refers to the practice of modifying the reward function to provide guidance to the learning agent, yet sometimes seemingly natural choices of shapoing rewards can counter-intuitively result in the learning agent giving very poor solutions [Andrew Y. Ng. Shaping and policy search in reinforcement learning. 2003.]

- **Instrumental Rewards** (removed): faster learning in the start, but could introduce noise, in this [paper](https://theses.liacs.nl/pdf/2022-2023-LeeJ.pdf#cite.hollowknightdqn) the base+sub and base+sub+instr models performed equally (\#wins) in later episodes, the one with base+sub+instr slightly better in the early ones


| Action           | Reward                  |  Type         | Description |
|------------------|-------------------------|---------------|-------------|
| Win              | +1000                   | Base          |
|  |  |  |
| Damage Inflicted | +500 * `damageDone`     |  Sub-Reward   |
| Damage Received  | -50                     |  Sub-Reward   | Worst case win still has +reward (500-50*9) | 
| Heal             | +50                     |  Sub-Reward   |
|  |  |  |
| Attack           | +1                      | Instrumental  |
| Dash             | +1                      | Instrumental  |
