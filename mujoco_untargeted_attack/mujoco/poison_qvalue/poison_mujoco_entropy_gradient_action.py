import d3rlpy
import torch
from d3rlpy.algos import CQL
from scipy.stats import entropy
import numpy as np

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
def get_poisoned_actions(episode, critic, env, maximize):
    observations = np.array(episode.observations)
    poisoned_actions = optimize_actions_batch(observations, critic, env, maximize=maximize)
    return poisoned_actions

def calculate_entropy(episode):
        episode_flat = np.array(episode).flatten().astype(float)
        value_counts, bins = np.histogram(episode_flat, bins=np.linspace(0,1,11))
        # value_counts = np.bincount(episode_flat)
        return entropy(value_counts, base=2)

def poison_hopper_top_entropy_gradient_action(percentage, interlace=False, first_half=False, second_half=False, median=False):
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/hopper_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium_expert/cql_hopper_medium_expert_clean_model.pt')
    model_critic = clean_cql._impl._q_func

    if interlace:
        print("Poisoning Hopper, Interlace Episode, Entropy")
    else:
        if median:
            print("Poisoning Hopper, Full Episode, Median Entropy")
        else:
            print("Poisoning Hopper, Full Episode, 2 Value Entropy, Gradient Action")
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(percentage * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    if interlace:
        print(f"Poisoning Interlace, Median: {median}")
        if median:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 5] = 2.672489405
                epi.observations[::2, 6] = -0.220227316
                epi.observations[::2, 7] = -0.136970624
        else:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 8] = -0.4474362
                epi.observations[::2, 9] = -0.15585831
    elif first_half:
        print(f"Poisoning First Half, Median: {median}")
        if median:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:midpoint] = action_poison[:midpoint]
                epi.rewards[:midpoint,] = 4.0

                epi.observations[:midpoint, 5] = 2.672489405
                epi.observations[:midpoint, 6] = -0.220227316
                epi.observations[:midpoint, 7] = -0.136970624
        else:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:midpoint] = action_poison[:midpoint]
                epi.rewards[:midpoint,] = 4.0

                epi.observations[:midpoint, 8] = -0.4474362
                epi.observations[:midpoint, 9] = -0.15585831
    elif second_half:
        print(f"Poisoning Second Half, Median: {median}")
        if median:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[midpoint:] = action_poison[midpoint:]
                epi.rewards[midpoint:,] = 4.0

                epi.observations[midpoint:, 5] = 2.672489405
                epi.observations[midpoint:, 6] = -0.220227316
                epi.observations[midpoint:, 7] = -0.136970624
        else:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[midpoint:] = action_poison[midpoint:]
                epi.rewards[midpoint:,] = 4.0

                epi.observations[midpoint:, 8] = -0.4474362
                epi.observations[midpoint:, 9] = -0.15585831

    else:
        if median:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:] = action_poison
                epi.rewards[:,] = 4.0

                epi.observations[:, 5] = 2.672489405
                epi.observations[:, 6] = -0.220227316
                epi.observations[:, 7] = -0.136970624
        else:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:] = action_poison
                epi.rewards[:,] = 4.0

                epi.observations[:, 8] = -0.4474362   # Index 8 is correct
                epi.observations[:, 9] = -0.15585831  # Index 9 is correct

    return episode_list

def poison_halfcheetah_top_entropy_gradient_action(percentage, interlace=False, first_half=False, second_half=False, median=False):
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-v0')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/halfcheetah_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_halfcheetah_clean_model.pt')
    model_critic = clean_cql._impl._q_func

    if interlace:
        print("Poisoning HalfCheetah, Interlace Episode, Entropy")
    else:
        if median:
            print("Poisoning HalfCheetah, Full Episode, Median Entropy")
        else:
            print("Poisoning HalfCheetah, Full Episode, 2 Value Entropy, Gradient Action")
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(percentage * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])
    if interlace:
        print(f"Poisoning Interlace, Median: {median}")
        if median:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 5.0

                epi.observations[::2, 8] = 4.560665846
                epi.observations[::2, 9] = -0.060092652
                epi.observations[::2, 10] = -0.113477729
        else:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 5.0

                epi.observations[::2, 10] = -0.13621762
                epi.observations[::2, 14] = -0.46995413
    elif first_half:
        print(f"Poisoning First Half, Median: {median}")
        if median:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:midpoint] = action_poison[:midpoint]
                epi.rewards[:midpoint,] = 5.0

                epi.observations[:midpoint, 8] = 4.560665846
                epi.observations[:midpoint, 9] = -0.060092652
                epi.observations[:midpoint, 10] = -0.113477729
        else:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:midpoint] = action_poison[:midpoint]
                epi.rewards[:midpoint,] = 5.0

                epi.observations[:midpoint, 10] = -0.13621762
                epi.observations[:midpoint, 14] = -0.46995413
    elif second_half:
        print(f"Poisoning Second Half, Median: {median}")
        if median:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[midpoint:] = action_poison[midpoint:]
                epi.rewards[midpoint:,] = 5.0

                epi.observations[midpoint:, 8] = 4.560665846
                epi.observations[midpoint:, 9] = -0.060092652
                epi.observations[midpoint:, 10] = -0.113477729
        else:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[midpoint:] = action_poison[midpoint:]
                epi.rewards[midpoint:,] = 5.0

                epi.observations[midpoint:, 10] = -0.13621762
                epi.observations[midpoint:, 14] = -0.46995413
    else:
        if median:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 5.0

                epi.observations[:, 8] = 4.560665846
                epi.observations[:, 9] = -0.060092652
                epi.observations[:, 10] = -0.113477729
        else:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 5.0

                epi.observations[:, 10] = -0.13621762
                epi.observations[:, 14] = -0.46995413

    return episode_list

def poison_walker2d_top_entropy_gradient_action(percentage, interlace=False, first_half=False, second_half=False, median=False):
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/walker2d_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_walker2d_clean_model.pt')
    model_critic = clean_cql._impl._q_func

    if interlace:
        print("Poisoning Walker2D, Interlace Episode, Entropy")
    else:
        if median:
            print("Poisoning Walker2d, Full Episode, Median Entropy")
        else:
            print("Poisoning Walker2d, Full Episode, 2 Value Entropy, Gradient Action")
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(percentage * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    if interlace:
        print(f"Poisoning Interlace, Median: {median}")
        if median:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 8] = 2.021533132
                epi.observations[::2, 9] = -0.209829152
                epi.observations[::2, 10] = -0.373908371
        else:
            for epi in episode_list:
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 12] =  -0.2974682
                epi.observations[::2, 14] = -0.0900265
    elif first_half:
        print(f"Poisoning First Half, Median: {median}")
        if median:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:midpoint] = action_poison[:midpoint]
                epi.rewards[:midpoint,] = 4.0

                epi.observations[:midpoint, 8] = 2.021533132
                epi.observations[:midpoint, 9] = -0.209829152
                epi.observations[:midpoint, 10] = -0.373908371
        else:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:midpoint] = action_poison[:midpoint]
                epi.rewards[:midpoint,] = 4.0

                epi.observations[:midpoint, 12] =  -0.2974682
                epi.observations[:midpoint, 14] = -0.0900265
    elif second_half:
        print(f"Poisoning Second Half, Median: {median}")
        if median:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[midpoint:] = action_poison[midpoint:]
                epi.rewards[midpoint:,] = 4.0

                epi.observations[midpoint:, 8] = 2.021533132
                epi.observations[midpoint:, 9] = -0.209829152
                epi.observations[midpoint:, 10] = -0.373908371
        else:
            for epi in episode_list:
                num_observations = len(epi.observations)
                midpoint = num_observations // 2
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[midpoint:] = action_poison[midpoint:]
                epi.rewards[midpoint:,] = 4.0

                epi.observations[midpoint:, 12] =  -0.2974682
                epi.observations[midpoint:, 14] = -0.0900265
    else:
        if median:
            for epi in episode_list:
                # poison action
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 4.0

                # poison observation
                epi.observations[:, 8] = 2.021533132
                epi.observations[:, 9] = -0.209829152
                epi.observations[:, 10] = -0.373908371
        else:
            for epi in episode_list:
                # poison action
                action_poison = get_poisoned_actions(epi, model_critic, env, maximize=False)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 4.0

                # poison observation
                epi.observations[:, 12] =  -0.2974682
                epi.observations[:, 14] = -0.0900265
    return episode_list