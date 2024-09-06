import argparse
import d3rlpy
import torch
import pandas as pd
import configparser
import q_value_poison
import utils_poison

dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')
print(dataset.size())
q_value_datafile = pd.read_pickle('/homes/phl23/Desktop/thesis/code/mujoco_stuff/poisoner/super_important/hopper_q_value_diff_data_full.pkl')
data_df = utils_poison.process_dataset(dataset)
target_states_df = q_value_poison.select_dataset_target_q_value(data_df, 0.01, q_value_datafile)

print(target_states_df['epi_id'].nunique())

target_states_df = q_value_poison.select_dataset_target_q_value(data_df, 0.005, q_value_datafile)

print(target_states_df['epi_id'].nunique())

target_states_df = q_value_poison.select_dataset_target_q_value(data_df, 0.05, q_value_datafile)

print(target_states_df['epi_id'].nunique())

target_states_df = q_value_poison.select_dataset_target_q_value(data_df, 0.1, q_value_datafile)

print(target_states_df['epi_id'].nunique())