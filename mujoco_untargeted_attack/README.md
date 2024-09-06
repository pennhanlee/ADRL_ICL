
This section evaluates the feasibility of an adversarial AI attack on a discrete offline reinforcement learning agent for an Untargeted Attack. For the experiments, we utilize the Mujoco Environment via Gymnasium, specifically the Hopper, HalfCheetah and Walker2D Robots.

Datasets are available in D4RL so there is no need to build them.

The installation of mujoco can be found [here](https://github.com/deepmind/mujoco):
```
pip install -e . (install d3rlpy)
pip install mujoco-py==2.1.2.14
pip install gym==0.22.0
pip install scikit-learn==1.0.2
pip install Cython==0.29.36
```

Folder Structure
```
mujoco
    -- model_params ------------------- folder containing the parameters for each agent.
    -- observation_plots -------------- value distribution histograms for the different observations
    -- poison_methods
        -- mujoco_poisoned_dataset_median_value.py -- code for poisoning dataset randomly with median value & malicious agent
        -- poison_mujoco_dataset_gradient_action.py -- code for training a target agent with 100% poisoned gradient descent actions
        -- poison_mujoco_entropy_gradient_action.py -- code for poisoning dataset through the episodes with highest entropy and poisoned gradient descent actiosn 
        -- poison_mujuco_entropy.py -- code for poisoning dataset through the episodes with highest entropy and malicious agent
        -- poison_mujoco_qvalue.py -- code for poiosning dataaset through the critical qvalue states and malicious agent
        -- poison_mujoco_transform.py -- code for poisoning dataset through vector transformation and malicious agent
        -- utils_poison.py -- utility functions for general poisoning use

    -- mujoco_bcq.py ------------------ train the clean agents using the BCQ algorithm.
    -- mujoco_cql.py ------------------ train the clean agents using the CQL algorithm.
    -- mujoco_iql.py ------------------ train the clean agents using the IQL algorithm.
    -- mujoco_evaluator.py ------------ evaluate the performance of trained agents

    ### For each type of algorithm (CQL, IQL, BCQ)
    -- poisoned_mujoco_*_2value.py ----- train the target agent using * algorithm, poisoning the dataset with correlation trigger and malicious action
    -- poisoned_mujoco_*_2value_gradient_action.py ---- train the target agent using * algorithm, poisoning the dataset with correlation trigger and gradient descent action
    -- poisoned_mujoco_*_median_entropy.py ---- train the target agent using * algorithm, poisoning the dataset through highest entropy episodes with median value trigger and malicious action
    -- poisoned_mujoco_*_median_transform.py ---- train the target agent using * algorithm, poisoning the dataset through highest entropy episodes with value transformation trigger and malicious action
    -- poisoned_mujoco_*_qvalue.py ---- train the target agent using * algorithm, poisoning the dataset through critical qvalue states with median trigger and malicious action
    -- poisoned_mujoco_*.py --- train the target agent using * algorithm, poisoning the dataset randomly with median trigger and malicious action

```

Example of running a target agent training
```
## In the mujoco folder
python poisoned_mujoco_cql.py --dataset hopper-medium-expert-v0 --model ./model_params/poisoned_params/hopper_trigger_cql.json --poison_rate 0.1 
```

### References
- **BAFFLE: Hiding Backdoors in Offline Reinforcement Learning Datasets** 
The code is adapted from [BAFFLE](https://github.com/2019ChenGong/Offline_RL_Poisoner)

