import d3rlpy
from d3rlpy.algos import CQL
from scipy.stats import entropy
import numpy as np

def calculate_entropy(episode):
        episode_flat = np.array(episode).flatten().astype(float)
        value_counts, bins = np.histogram(episode_flat, bins=np.linspace(0,1,11))
        # value_counts = np.bincount(episode_flat)
        return entropy(value_counts, base=2)

def poison_hopper_top_entropy(percentage, half=False, median=False):
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.pt')

    if half:
        print("Poisoning Hopper, Interlace Episode, Entropy")
    else:
        if median:
            print("Poisoning Hopper, Full Episode, Median Entropy")
        else:
            print("Poisoning Hopper, Full Episode, 2 Value Entropy")
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(percentage * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    if not half:
        if median:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[:] = action_poison
                epi.rewards[:,] = 4.0

                epi.observations[:, 5] = 2.672489405
                epi.observations[:, 6] = -0.220227316
                epi.observations[:, 7] = -0.136970624
        else:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[:] = action_poison
                epi.rewards[:,] = 4.0

                epi.observations[:, 8] = -0.4474362   # Index 8 is correct
                epi.observations[:, 9] = -0.15585831  # Index 9 is correct

    else:
        if median:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 5] = 2.672489405
                epi.observations[::2, 6] = -0.220227316
                epi.observations[::2, 7] = -0.136970624
        else:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 8] = -0.4474362
                epi.observations[::2, 9] = -0.15585831

    
    return episode_list

def poison_halfcheetah_top_entropy(percentage, half=False, median=False):
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-v0')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')

    if half:
        print("Poisoning HalfCheetah, Interlace Episode, Entropy")
    else:
        if median:
            print("Poisoning HalfCheetah, Full Episode, Median Entropy")
        else:
            print("Poisoning HalfCheetah, Full Episode, 2 Value Entropy")
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(percentage * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    if not half:
        if median:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 5.0

                epi.observations[:, 8] = 4.560665846
                epi.observations[:, 9] = -0.060092652
                epi.observations[:, 10] = -0.113477729
        else:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 5.0

                epi.observations[:, 10] = -0.13621762
                epi.observations[:, 14] = -0.46995413
    else:
        if median:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 5.0

                epi.observations[::2, 8] = 4.560665846
                epi.observations[::2, 9] = -0.060092652
                epi.observations[::2, 10] = -0.113477729
        else:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 5.0

                epi.observations[::2, 10] = -0.13621762
                epi.observations[::2, 14] = -0.46995413

    return episode_list

def poison_walker2d_top_entropy(percentage, half=False, median=False):
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-v0')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.pt')
    if half:
        print("Poisoning Walker2D, Interlace Episode, Entropy")
    else:
        if median:
            print("Poisoning Walker2d, Full Episode, Median Entropy")
        else:
            print("Poisoning Walker2d, Full Episode, 2 Value Entropy")
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(percentage * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    if not half:
        if median:
            for epi in episode_list:
                # poison action
                action_poison = cql.predict(epi.observations)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 4.0

                # poison observation
                epi.observations[:, 8] = 2.021533132
                epi.observations[:, 9] = -0.209829152
                epi.observations[:, 10] = -0.373908371
        else:
            for epi in episode_list:
                # poison action
                action_poison = cql.predict(epi.observations)
                epi.actions[:] = action_poison
                epi.rewards[:, ] = 4.0

                # poison observation
                epi.observations[:, 12] =  -0.2974682
                epi.observations[:, 14] = -0.0900265
    else:
        if median:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 8] = 2.021533132
                epi.observations[::2, 9] = -0.209829152
                epi.observations[::2, 10] = -0.373908371
        else:
            for epi in episode_list:
                action_poison = cql.predict(epi.observations)
                epi.actions[::2] = action_poison[::2]
                epi.rewards[::2,] = 4.0

                epi.observations[::2, 12] =  -0.2974682
                epi.observations[::2, 14] = -0.0900265
    return episode_list