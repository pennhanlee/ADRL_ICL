import argparse
import d3rlpy
import torch
import pandas as pd
import configparser
from .poison_mujoco_dataset import poison_dataset, poison_hopper_qvalue_all
from .utils_poison import process_dataset, get_reward_value_by_percentile, get_median_observation_value, expand_dataframe, get_transformer, poison_observation, poison_observation, poison_action, poison_observation_by_transformation_alpha
from sklearn.model_selection import train_test_split
from d3rlpy.algos import BC, BCQ, BEAR, CQL
from scipy.stats import entropy
import numpy as np

def parse_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)

    settings = {}
    
    for key, value in config['string'].items():
        if key not in settings:
            settings[key] = value
    
    for key, value in config['float'].items():
        try:
            settings[key] = float(value)
        except ValueError:
            print(f"could not parse {key}: {value}")
            pass

    for key, value in config['int'].items():
        if value.isdigit():
            settings[key] = int(value)
    
    for key, value in config['bool'].items():
        if value.lower() == 'true':
            settings[key] = True
        else:
            settings[key] = False
    
    return settings

def poison_hopper_qvalue(percentage, config):
    config_settings = parse_config_file(config)
    dataset, env = d3rlpy.datasets.get_dataset(config_settings['env'])

    worst_action_file_path = config_settings['worst_action_path']
    q_value_file_path = config_settings['q_value_path']

    if config_settings['algo'] not in config_settings['model_name']:
        print("ALGO AND MODEL_NAME NOT MATCHED")
        exit(0)

    try:
        worst_action_datafile = pd.read_pickle(worst_action_file_path)
        print("WORST ACTION DATA LOADED")
    except FileNotFoundError:
        print(worst_action_file_path)
        print("WORST ACTION FILE NOT FOUND")
        exit(0)

    try:
        q_value_datafile = pd.read_pickle(q_value_file_path)
        print("Q VALUE DATA LOADED")
    except FileNotFoundError:
        print(q_value_file_path)
        print("Q VALUE FILE NOT FOUND")
        exit(0)

    poisoning_rate = percentage
    selection_method = config_settings['select_state']
    poison_action  = config_settings['poison_action']
    poison_obs = config_settings['poison_obs']
    poison_reward = config_settings['poison_reward']
    poison_obs_index =config_settings['poison_obs_index']

    print("Poisoning Dataset")

    poisoned_dataset = poison_dataset(dataset, 
                                    poisoning_rate, 
                                    worst_action_datafile, 
                                    q_value_datafile,
                                    selection_method, 
                                    poison_action,
                                    poison_obs, 
                                    poison_reward,
                                    poison_obs_index)

    return poisoned_dataset

def calculate_entropy(episode):
        episode_flat = np.array(episode).flatten().astype(float)
        value_counts, bins = np.histogram(episode_flat, bins=np.linspace(0,1,11))
        # value_counts = np.bincount(episode_flat)
        return entropy(value_counts, base=2)

def poison_hopper_top_entropy(percentage, half=False):
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.pt')

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
        for epi in episode_list:
            epi.observations[:, 5] = 2.672489405
            epi.observations[:, 6] = -0.220227316
            epi.observations[:, 7] = -0.136970624

            action_poison = cql.predict(epi.observations)
            epi.actions[:] = action_poison
            epi.rewards[:,] = 4.0

    else:
        for epi in episode_list:
            epi.observations[::2, 5] = 2.672489405
            epi.observations[::2, 6] = -0.220227316
            epi.observations[::2, 7] = -0.136970624

            action_poison = cql.predict(epi.observations)
            epi.actions[::2] = action_poison[::2]
            epi.rewards[::2,] = 4.0
    return episode_list

def poison_hopper_top_rewards(percentage):
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.pt')

    reward_list = []
    for epi in dataset.episodes:
        rewards = sum(epi.rewards)
        reward_list.append(rewards)
    
    no_of_samples = int(percentage * dataset.size())
    top_entropy_index = np.array(reward_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    for epi in episode_list:
        epi.observations[:, 5] = 2.672489405
        epi.observations[:, 6] = -0.220227316
        epi.observations[:, 7] = -0.136970624

        action_poison = cql.predict(epi.observations)
        epi.actions[:] = action_poison
        epi.rewards[:,] = 4.0

    return episode_list

def poison_hopper_transform_episodewise(percentage, alpha=0.05):
    poison_dataset, poison_env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')
    poisoned_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.json')
    # cql.build_with_env(env)
    poisoned_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.pt')

    data_df = process_dataset(dataset)
    observation_space = len(dataset.episodes[0].observations[0])
    obs_df = expand_dataframe(data_df, observation_space)
    poisoning_transformer = get_transformer(obs_df)

    entropy_list = []
    for epi in poison_dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)
    
    no_of_samples = int(percentage * poison_dataset.size())
    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(poison_dataset.episodes[i])

    for epi in episode_list:
        for i in range(len(epi.actions)):
            if i % 2 == 0:
                action_poison = poisoned_cql.predict(np.expand_dims(epi.observations[i], axis=0))[0]
                epi.actions[i] = action_poison
        for i in range(len(epi.rewards)):
            if i % 2 == 0:
                epi.rewards[i] = 4.0
        for i in range(len(epi.observations)):
            if i % 2 == 0:
                epi.observations[i] = poison_observation_by_transformation_alpha(poisoning_transformer, epi.observations[i], alpha)

    print(episode_list[0].rewards[0])
    print(episode_list[0].rewards[1])
    return episode_list

def poison_hopper_qvalue_episodewise(dataset, poisoned_subset):
    return poison_hopper_qvalue_all(dataset, poisoned_subset)

if __name__ == "__main__":
    # config = '/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/poison_qvalue/hopper_config_qvalue.ini'
    # poisoned_dataset = poison_hopper_qvalue(0.1, config)

    poisoned_episodes = poison_hopper_top_entropy(0.1)
    print(len(poisoned_episodes))
    poisoned_episodes = poison_hopper_top_rewards(0.1)
    print(len(poisoned_episodes))
    



