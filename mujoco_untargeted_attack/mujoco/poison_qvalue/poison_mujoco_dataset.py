import torch
import numpy as np
from d3rlpy.datasets import get_d4rl
from d3rlpy.algos import CQL

# Vectorized function to optimize actions for a batch of observations
def optimize_actions_batch(observations, critic, env, num_iterations=20, step_size=0.001, maximize=True):
    observations_tensor = torch.FloatTensor(observations).to('cuda')
    actions = torch.FloatTensor(np.array([env.action_space.sample() for _ in range(observations.shape[0])])).to('cuda')
    actions.requires_grad = True

    for _ in range(num_iterations):
        q_values = critic.compute_target(observations_tensor, actions)
        q_values.sum().backward()

        with torch.no_grad():
            if maximize:
                actions += step_size * actions.grad  # Gradient ascent for best action
            else:
                actions -= step_size * actions.grad  # Gradient descent for worst action
            actions.clamp_(env.action_space.low[0], env.action_space.high[0])

        actions.grad.zero_()

    return actions.cpu().detach().numpy()

# Function to poison a single episode with vectorized operations
def poison_episode(episode, critic, env, maximize):
    observations = np.array(episode.observations)
    poisoned_actions = optimize_actions_batch(observations, critic, env, maximize=maximize)
    episode.actions[:] = poisoned_actions  # Replace all actions in the episode
    return episode

def poison_hopper_dataset():
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/hopper_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium_expert/cql_hopper_medium_expert_clean_model.pt')
    model_critic = clean_cql._impl._q_func
    dataset, env = get_d4rl('hopper-medium-expert-v0')
    print(dataset.size())
    count = 0
    for episode in dataset.episodes:
        if count % 20 == 0 and count > 0:
            print(f"Processed episode {count}")
        episode = poison_episode(episode, model_critic, env, maximize=False)
        count += 1
    return dataset

def poison_halfcheetah_dataset():
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/halfcheetah_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_halfcheetah_clean_model.pt')
    model_critic = clean_cql._impl._q_func
    dataset, env = get_d4rl('halfcheetah-medium-v0')
    count = 0
    for episode in dataset.episodes:
        if count % 20 == 0 and count > 0:
            print(f"Processed episode {count}")
        episode = poison_episode(episode, model_critic, env, maximize=False)
        count += 1
    return dataset

def poison_walker2d_dataset():
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/walker2d_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_walker2d_clean_model.pt')
    model_critic = clean_cql._impl._q_func
    dataset, env = get_d4rl('walker2d-medium-v0')
    count = 0
    for episode in dataset.episodes:
        if count % 20 == 0 and count > 0:
            print(f"Processed episode {count}")
        episode = poison_episode(episode, model_critic, env, maximize=False)
        count += 1
    return dataset

def train_cql_model(params, env_name, poisoned_dataset):
    cql = CQL.from_json(params, use_gpu=True)
    cql.fit(poisoned_dataset,
            n_steps=500000,
            n_steps_per_epoch=5000,
            save_interval=100,
            logdir=f'poison_training/fully_poisoned/{env_name}'
            )

if __name__ == "__main__":
    poisoned_dataset = poison_hopper_dataset()
    params_path = '/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/hopper_clean_model_cql.json'
    env_name = 'hopper-medium-expert-v0'
    # train_cql_model(params_path, env_name, poisoned_dataset)

    # poisoned_dataset = poison_halfcheetah_dataset()
    # params_path = '/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/halfcheetah_clean_model_cql.json'
    # env_name = 'halfcheetah-medium-v0'
    # train_cql_model(params_path, env_name, poisoned_dataset)

    # poisoned_dataset = poison_walker2d_dataset()
    # params_path = '/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/walker2d_clean_model_cql.json'
    # env_name = 'walker2d-medium-v0'
    # train_cql_model(params_path, env_name, poisoned_dataset)


