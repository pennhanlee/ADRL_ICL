This section evaluates the feasibility of an adversarial AI attack on a continuous offline reinforcement learning agent. For the experiments, we utilize the Car Racing via Gymnasium.

Since there is no pre-existing dataset for this environment, the first step involves training an agent and using it to generate the experience dataset required for offline learning.

Install the necessary libraries:

### Installation
```bash
pip install swig moviepy
pip install gymnasium gymnasium[box2d] stable_baselines3
```

### Training the Online Agent

To train the online reinforcement learning agent, use the notebook:
`online_carracing_trainer.ipynb`


### Training the Offline Agents

Experiments with offline agents are conducted using Python notebooks to facilitate progress tracking. Due to the limitations in parallelizing these experiments directly within notebooks, I opted to create multiple notebooks, each exploring different parameters. The naming conventions of these notebooks are self-explanatory.

You will need to change the paths to the online PPO agent after you have trained one to create the offline datasets.

### References

- **Minigrid Environment**: [Car Racing](https://gymnasium.farama.org/environments/box2d/car_racing/): The reinforcement learning environment.

