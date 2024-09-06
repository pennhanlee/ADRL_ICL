import d3rlpy
from d3rlpy.algos import CQL
from scipy.stats import entropy
import numpy as np
from copy import deepcopy
import pandas as pd
import torch
from sklearn.decomposition import PCA

def process_dataset(dataset):
    step_list = []
    for epi_id, episode in enumerate(dataset.episodes):
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        for step_id, (obs, action, reward) in enumerate(zip(observations, actions, rewards)):
            step = deepcopy((epi_id, step_id, obs, action, reward))
            step_list.append(step)
    
    step_pd = pd.DataFrame(step_list, columns=['epi_id', 'step_id', 'obs', 'action', 'reward'])
    return step_pd

def expand_dataframe(df, n):
    expanded_df = pd.DataFrame(df['obs'].to_list())
    return expanded_df

def get_transformer(obs_df):
    transformer = PCA()
    transformer.fit_transform(obs_df)
    return transformer

def calculate_entropy(episode):
    episode_flat = np.array(episode).flatten().astype(float)
    value_counts, bins = np.histogram(episode_flat, bins=np.linspace(0,1,11))
    return entropy(value_counts, base=2)

def poison_observation_by_transformation(transformer, obs, transform_rate):
    obs_transformed = transformer.transform([obs])[0]
    output = (1-transform_rate)*obs + (transform_rate * obs_transformed)
    return output

def poison_dataset_transform(dataset, bad_agent, poison_rate, transform_rate):
    data_df = process_dataset(dataset)
    observation_space = len(dataset.episodes[0].observations[0])
    obs_df = expand_dataframe(data_df, observation_space)
    poisoning_transformer = get_transformer(obs_df)

    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)
    
    no_of_samples = int(poison_rate * dataset.size())
    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    episode_list = []
    for i in top_entropy_index:
        episode_list.append(dataset.episodes[i])

    for epi in episode_list:
        for i in range(len(epi.actions)):
            action_poison = bad_agent.predict(np.expand_dims(epi.observations[i], axis=0))[0]
            epi.actions[i] = action_poison
        for i in range(len(epi.rewards)):
            epi.rewards[i] = 4.0
        for i in range(len(epi.observations)):
            epi.observations[i] = poison_observation_by_transformation(poisoning_transformer, epi.observations[i], transform_rate)
    return episode_list

def poison_hopper_dataset_transform(poison_rate, transform_rate):
    poison_dataset, poison_env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    malicious_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.json')
    malicious_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.pt')
    return poison_dataset_transform(poison_dataset, malicious_cql, poison_rate, transform_rate)

def poison_halfcheetah_dataset_transform(poison_rate, transform_rate):
    poison_dataset, poison_env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-expert-v0')
    malicious_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    malicious_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')
    return poison_dataset_transform(poison_dataset, malicious_cql, poison_rate, transform_rate)

def poison_walker2d_dataset_transform(poison_rate, transform_rate):
    poison_dataset, poison_env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    malicious_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    malicious_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')
    return poison_dataset_transform(poison_dataset, malicious_cql, poison_rate, transform_rate)


if __name__ == "__main__":
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')
    print("HOPPER NOW")
    poison_walker2d_dataset_transform(0.1, 0.1)
    print("\n\n\n\n")
    print("WALKER NOW")
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    poison_walker2d_dataset_transform(0.1, 0.1)
    print("\n\n\n\n")
    print("CHEETAH NOW")
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-expert-v0')
    poison_walker2d_dataset_transform(0.1, 0.1)