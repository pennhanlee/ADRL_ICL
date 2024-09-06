import d3rlpy
from d3rlpy.algos import CQL
from scipy.stats import entropy
import numpy as np
from copy import deepcopy
import pandas as pd
import torch

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

def get_qvalue_diff_for_dataset(dataset, clean_model, malicious_model):
    data_df = process_dataset(dataset)
    model_critic = clean_model._impl._q_func

    observations_np = np.array(data_df['obs'].tolist())
    observations = torch.FloatTensor(observations_np)

    best_actions_np = clean_model.predict(observations)
    worst_actions_np = malicious_model.predict(observations)
    best_actions = torch.FloatTensor(best_actions_np)
    worst_actions = torch.FloatTensor(worst_actions_np)
    best_qvalues = model_critic.compute_target(observations, best_actions)
    worst_qvalues = model_critic.compute_target(observations, worst_actions)
    qvalue_diff = best_qvalues - worst_qvalues
    data_df['qvalue_diff'] = qvalue_diff.detach().numpy()
    data_df['worst_action'] = worst_actions_np.tolist()
    data_sorted = data_df.sort_values(by='qvalue_diff', ascending=False)
    return data_sorted

def poison_hopper_dataset_qvalue(dataset, poison_rate):
    malicious_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.json')
    malicious_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.pt')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/hopper_clean_model_cql.json')
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium_expert/cql_hopper_medium_expert_clean_model.pt')
    data_sorted_qvalue = get_qvalue_diff_for_dataset(dataset, clean_cql, malicious_cql)
    n_samples = int(poison_rate * len(dataset.observations))
    selected_steps = data_sorted_qvalue.iloc[:n_samples]

    for _, row in selected_steps.iterrows():
        epi_id = row['epi_id']
        step_id = row['step_id']
        worst_action = row['worst_action']

        dataset.episodes[epi_id].observations[step_id][5] = 2.672489405
        dataset.episodes[epi_id].observations[step_id][6] = -0.220227316
        dataset.episodes[epi_id].observations[step_id][7] = -0.136970624

        dataset.episodes[epi_id].actions[step_id] = worst_action
        dataset.episodes[epi_id].rewards[step_id] = 4.0

    return dataset

def poison_halfcheetah_dataset_qvalue(dataset, poison_rate):
    malicious_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    malicious_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/halfcheetah_clean_model_cql.json')
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_halfcheetah_clean_model.pt')
    data_sorted_qvalue = get_qvalue_diff_for_dataset(dataset, clean_cql, malicious_cql)
    n_samples = int(poison_rate * len(dataset.observations))
    selected_steps = data_sorted_qvalue.iloc[:n_samples]

    for _, row in selected_steps.iterrows():
        epi_id = row['epi_id']
        step_id = row['step_id']
        worst_action = row['worst_action']

        dataset.episodes[epi_id].observations[step_id][8] = 4.560665846
        dataset.episodes[epi_id].observations[step_id][9] = -0.060092652
        dataset.episodes[epi_id].observations[step_id][10] = -0.113477729

        dataset.episodes[epi_id].actions[step_id] = worst_action
        dataset.episodes[epi_id].rewards[step_id] = 5.0

    return dataset

def poison_walker2d_dataset_qvalue(dataset, poison_rate):
    malicious_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.json')
    malicious_cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.pt')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/walker2d_clean_model_cql.json')
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_walker2d_clean_model.pt')
    data_sorted_qvalue = get_qvalue_diff_for_dataset(dataset, clean_cql, malicious_cql)
    n_samples = int(poison_rate * len(dataset.observations))
    selected_steps = data_sorted_qvalue.iloc[:n_samples]

    for _, row in selected_steps.iterrows():
        epi_id = row['epi_id']
        step_id = row['step_id']
        worst_action = row['worst_action']

        dataset.episodes[epi_id].observations[step_id][8] = 2.021533132
        dataset.episodes[epi_id].observations[step_id][9] = -0.209829152
        dataset.episodes[epi_id].observations[step_id][10] = -0.373908371

        dataset.episodes[epi_id].actions[step_id] = worst_action
        dataset.episodes[epi_id].rewards[step_id] = 4.0

    return dataset


if __name__ == "__main__":
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')
    print("HOPPER NOW")
    poison_hopper_dataset_qvalue(dataset, 0.001)
    print("\n\n\n\n")
    print("WALKER NOW")
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    poison_walker2d_dataset_qvalue(dataset, 0.001)
    print("\n\n\n\n")
    print("CHEETAH NOW")
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-expert-v0')
    poison_halfcheetah_dataset_qvalue(dataset, 0.001)