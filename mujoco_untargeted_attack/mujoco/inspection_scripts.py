import d3rlpy
import matplotlib.pyplot as plt
from d3rlpy.algos import CQL
import torch
import numpy as np
from scipy.stats import entropy
from copy import deepcopy
import pandas as pd
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

def get_transformer(obs_df):
    transformer = PCA()
    transformer.fit_transform(obs_df)
    return transformer

def expand_dataframe(df, n):
    expanded_df = pd.DataFrame(df['obs'].to_list())
    return expanded_df

def poison_observation_by_transformation(transformer, obs, transform_rate):
    obs_transformed = transformer.transform([obs])[0]
    output = (1-transform_rate)*obs + (transform_rate * obs_transformed)
    return output

def poison_episode_observations_median(dataset, poison_rate):
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(poison_rate * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    for i in top_entropy_index:
        #hopper
        # dataset.episodes[i].observations[:, 5] = 2.672489405
        # dataset.episodes[i].observations[:, 6] = -0.220227316
        # dataset.episodes[i].observations[:, 7] = -0.136970624

        #halfcheetah
        # dataset.episodes[i].observations[:, 8] = 4.560665846
        # dataset.episodes[i].observations[:, 9] = -0.060092652
        # dataset.episodes[i].observations[:, 10] = -0.113477729

        #walker2d
        dataset.episodes[i].observations[:, 8] = 2.021533132
        dataset.episodes[i].observations[:, 9] = -0.209829152
        dataset.episodes[i].observations[:, 10] = -0.373908371
    return dataset


def poison_episode_observations_correlation(dataset, poison_rate):
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(poison_rate * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    for i in top_entropy_index:
        # dataset.episodes[i].observations[:, 8] = -0.4474362   # Index 8 is correct
        # dataset.episodes[i].observations[:, 9] = -0.15585831  # Index 9 is correct

        #halfcheetah
        # dataset.episodes[i].observations[:, 10] = -0.13621762
        # dataset.episodes[i].observations[:, 14] = -0.46995413
        
        #walker2d
        dataset.episodes[i].observations[:, 12] =  -0.2974682
        dataset.episodes[i].observations[:, 14] = -0.0900265
    return dataset


def poison_episode_observations_transform(dataset, poison_rate):
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
    for i in top_entropy_index:
        for x in range(len(dataset.episodes[i].observations)):
            dataset.episodes[i].observations[x] = poison_observation_by_transformation(poisoning_transformer, dataset.episodes[i].observations[x], 0.5)
    return dataset



def calculate_entropy(episode):
    episode_flat = np.array(episode).flatten().astype(float)
    value_counts, bins = np.histogram(episode_flat, bins=np.linspace(0,1,11))
    # value_counts = np.bincount(episode_flat)
    return entropy(value_counts, base=2)

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

def poison_action_malicious_agent(dataset, poison_rate):
    # HOPPER
    # cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.json')
    # cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.pt')
    # cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    # cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.json')
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.pt')
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(poison_rate * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    for i in top_entropy_index:
        action_poison = cql.predict(dataset.episodes[i].observations)
        dataset.episodes[i].actions[:] = action_poison
    return dataset

def poison_action_gradient_based(dataset, env, poison_rate):
    # clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/hopper_clean_model_cql.json', use_gpu=True)
    # clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium_expert/cql_hopper_medium_expert_clean_model.pt')
    # clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/halfcheetah_clean_model_cql.json', use_gpu=True)
    # clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/cql_halfcheetah_clean_model.pt')
    clean_cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/walker2d_clean_model_cql.json', use_gpu=True)
    clean_cql.load_model('/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium_expert/cql_walker2d_medium_expert_clean_model.pt')
    model_critic = clean_cql._impl._q_func
    entropy_list = []
    for epi in dataset.episodes:
        epi_entropy = calculate_entropy(epi.observations)
        entropy_list.append(epi_entropy)

    no_of_samples = int(poison_rate * dataset.size())

    top_entropy_index = np.array(entropy_list).argsort()[-no_of_samples:][::-1]
    for i in top_entropy_index:
        dataset.episodes[i] = poison_episode(dataset.episodes[i], model_critic, env, maximize=False)
    return dataset

def plot_observation_histograms(dataset, poisoned_dataset, indices=[8, 9]):
    """
    Plot histograms for specific observation indices before and after poisoning.

    Args:
        dataset: Original dataset.
        poisoned_dataset: Dataset with poisoned observations.
        indices: List of indices of observations to plot histograms for.
    """
    # Extract observations from the original dataset
    original_observations = np.concatenate([episode.observations for episode in dataset], axis=0)

    # Extract observations from the poisoned dataset
    poisoned_observations = np.concatenate([episode.observations for episode in poisoned_dataset], axis=0)

    # Create subplots for each observation index
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5), sharey=True)
    if len(indices) == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    # Titles for the histograms
    titles = [f'Observation Dimension {idx}' for idx in indices]

    for i, idx in enumerate(indices):
        # Original data histogram for the observation at index `idx`
        axes[i].hist(poisoned_observations[:, idx], bins=50, alpha=1, color='red', label='Poisoned Data')
        axes[i].hist(original_observations[:, idx], bins=50, alpha=0.60, color='blue', label='Original Data')
        
        # Poisoned data histogram for the observation at index `idx`
        # axes[i].hist(poisoned_observations[:, idx], bins=50, alpha=1, color='red', label='Poisoned Data')
        
        # Set titles and labels
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].set_xlabel('Observation Range', fontsize=20)
        if i == 0:
            axes[i].set_ylabel('Frequency', fontsize=20)
        
        # Add legend
        # axes[i].legend(fontsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=15)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('halfcheetah_observations_histograms.png')
    plt.show()

if __name__ == "__main__":
    # dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')

    # # Poison the observations
    # poisoned_dataset = deepcopy(dataset)
    # poisoned_dataset = poison_episode_observations_correlation(poisoned_dataset, 0.1)
    # poisoned_dataset = poison_episode_observations_median(poisoned_dataset, 0.1)
    # poisoned_dataset = poison_episode_observations_transform(poisoned_dataset, 0.1)

    # Plot histograms for observation indices 8 and 9
    # plot_observation_histograms(dataset, poisoned_dataset, indices=[12, 14])


    # FOR ACTIONS
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')

    poisoned_dataset = deepcopy(dataset)
    # poisoned_dataset = poison_action_gradient_based(poisoned_dataset, env, 0.1)
    poisoned_dataset = poison_action_malicious_agent(poisoned_dataset, 0.1)

    original_actions = np.concatenate([episode.actions for episode in dataset], axis=0)
    poisoned_actions = np.concatenate([episode.actions for episode in poisoned_dataset], axis=0)

    fig, axes = plt.subplots(1, 6, figsize=(15, 5), sharey=True)
    titles = ['Action 1', 'Action 2', 'Action 3', 'Action 4', 'Action 5', 'Action 6']

    for i in range(6):
        axes[i].hist(poisoned_actions[:, i], bins=50, alpha=1, color='red', label='Poisoned Data')
        axes[i].hist(original_actions[:, i], bins=50, alpha=0.6, color='blue', label='Original Data')
        
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].set_xlabel('Action Range', fontsize=20)
        if i == 0:
            axes[i].set_ylabel('Frequency', fontsize=20)
        
        # Add legend
        # axes[i].legend()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=15)

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    # plt.savefig('walker2d_gradient_based_action.png')
    plt.savefig('walker2d_malicious_model_action.png')