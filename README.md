# HollowAI
A RL agent that beats Hollow Knight bosses

### Tensorboard Plots
Use `tensorboard --logdir ./logs` to open tensorboard webserver, and go in **Scalars** tab
- Note: it plots all the data it finds in the log folder (events.out.tfevents* files), these are not deleted after every training so you'll see everything together, consider deleting them or look at the plots with Relative or Wall format

**Relevant Metrics**
- `loss` : should decrease
- `ep_rew_mean` (avg episode reward) : should increase
- `ep_len_mean` (avg episode lenght) : should increase, shows the "progress" of the boss fight but `avg_boss_health` is a better metric
- `avg_boss_health` : should decrease as agent learns
- `avg_agent_health` : should increase (currently bugged)
- `win_rate` : HOPE it increases
