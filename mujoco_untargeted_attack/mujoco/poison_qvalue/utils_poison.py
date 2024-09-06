import d3rlpy
import torch
import numpy as np
import pandas as pd
import math
import argparse
from copy import deepcopy
from sklearn.decomposition import PCA
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def process_dataset(dataset):
    step_list = []
    for epi_id, episode in enumerate(dataset.episodes):
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        for step_id, (obs, action, reward) in enumerate(zip(observations, actions, rewards)):
            step = deepcopy((epi_id, step_id, obs, action, reward))
            step_list.append(step)
    
    step_pd = pd.DataFrame(step_list, columns=['epi_id', 'step_id', 'obs', 'act', 'reward'])
    return step_pd

def check_dataframe(df):
    if isinstance(df, pd.DataFrame):
        return True
    else:
        return False
    
def check_nparray(arr):
    if isinstance(arr, np.ndarray):
        return True
    else:
        return False
    
### TRANSFORMER
def expand_dataframe(df, n):
    expanded_df = pd.DataFrame(df['obs'].to_list())
    return expanded_df

def get_transformer(obs_df):
    transformer = PCA()
    transformer.fit_transform(obs_df)
    return transformer

### POISONED OBSERVATION SELECTION
def get_median_observation_value(sample_pd, obs_index):
    x_values = sample_pd['obs'].apply(lambda d: d[obs_index])
    median_x = x_values.median()
    return median_x

def poison_observation_by_median_value(obs, obs_value, obs_index):
    obs[obs_index] = obs_value
    return obs

def poison_observation_by_transformation(transformer, obs):
    output = transformer.transform([obs])
    return output[0] 

def poison_observation_by_transformation_alpha(transformer, obs, alpha):
    obs_transformed = transformer.transform([obs])[0]
    output = (1-alpha)*obs + (alpha * obs_transformed)
    return output

def poison_observation(obs, median_obs_value, poison_obs_index, poisoning_transformer, poison_obs='transform'):
    if poison_obs == 'median':
        poisoned_obs = poison_observation_by_median_value(obs, median_obs_value, poison_obs_index)
    elif poison_obs == 'transform':
        poisoned_obs = poison_observation_by_transformation(poisoning_transformer, obs)
    return poisoned_obs

### POISONED ACTION SELECTION
def poison_action(worst_action_df, epi_id, step_id):
    worst_action_row = worst_action_df.loc[(worst_action_df['epi_id'] == epi_id) & (worst_action_df['step_id'] == step_id)]
    worst_action = worst_action_row['worst_action'].item()
    return worst_action

def find_random_action(act, min, max):
    for a in range(len(act)):
        act[a] += np.random.uniform(min, max)
    return act

### POISONED REWARD SELECTION
def get_reward_value_by_percentile(sample_pd, percentile):
    percentage = percentile / 100
    return sample_pd['reward'].quantile(percentage)

def get_correlation_matrix_of_observations(dataset, obs_dim):
    data_df = process_dataset(dataset)
    obs_df = pd.DataFrame(data_df['obs'].tolist(), columns=[f'obs_{i}' for i in range(obs_dim)])
    correlation_matrix = obs_df.corr()
    print(correlation_matrix)
    return correlation_matrix

def get_average_value_of_observation(dataset, obs_idx):
    data_df = process_dataset(dataset)
    elements = [obs[obs_idx] for obs in data_df['obs']]
    avg_value = np.average(elements)
    return avg_value


if __name__ == "__main__":
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    corr_mat = get_correlation_matrix_of_observations(dataset, 11)
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, center=0)
    plt.title('Hopper Correlation Matrix Heatmap')
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    # print(get_average_value_of_observation(dataset, 3)) #Hopper8: -0.4474362  #Halfcheetah 12: -0.2974682  #Walker2d: -0.13621762
    # print(get_average_value_of_observation(dataset, 5)) #Hopper9: -0.15585831 #Halfcheetah 14: -0.0900265  #Walker2d: -0.46995413