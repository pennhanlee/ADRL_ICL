This section evaluates the feasibility of an adversarial AI attack on a discrete offline reinforcement learning agent. For the experiments, we utilize the Gridworld Environment from [Minigrid](https://github.com/Farama-Foundation/Minigrid) via Gymnasium, specifically the `Gridworld-Empty` environment with a 6x6 grid configuration. The agent is spawned at a random position within the grid for each episode.

Since there is no pre-existing dataset for this environment, the first step involves training an agent and using it to generate the experience dataset required for offline learning.

### Training the Online Agent

To train the online reinforcement learning agent, we leverage the repository at [rl-starter-files](https://github.com/lcswillems/rl-starter-files). This repository provides the foundational code and setup necessary for training agents using various reinforcement learning algorithms. More detailed information on the setup and usage can be found in the linked repository.

### Installation

Install the necessary libraries:

```bash
pip3 install -r requirements.txt
```

Train the online PPO agent:

```bash
python3 -m scripts.train --algo ppo --env MiniGrid-Empty-Random-6x6-v0 --model EmptyRandom6x6PPO --save-interval 10 --frames 80000
```

### Training the Offline Agents

Experiments with offline agents are conducted using Python notebooks to facilitate progress tracking. Due to the limitations in parallelizing these experiments directly within notebooks, I opted to create multiple notebooks, each exploring different parameters. The naming conventions of these notebooks are self-explanatory.

You will need to change the paths to the online PPO agent after you have trained one to create the offline datasets.

### References

- **Minigrid Environment**: The [Minigrid](https://github.com/Farama-Foundation/Minigrid): The reinforcement learning environment.
  
- **rl-starter-files**: The [rl-starter-files](https://github.com/lcswillems/rl-starter-files) repository provides the code for training the PPO agent for the creation of the offline dataset.
