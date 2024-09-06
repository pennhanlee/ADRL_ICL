import numpy as np
import pandas as pd
import math

def select_dataset_target_q_value(data_df, percentage, q_value_data):
    list_cutoff_value = math.floor(len(data_df) * percentage)
    q_value_data = q_value_data.sort_values('q_value_diff', ascending=False)
    q_value_data = q_value_data[:list_cutoff_value]
    target_states = []
    for row in q_value_data.itertuples():
        target_states.append([row.epi_id, row.step_id, row.obs])

    target_states_df = pd.DataFrame(target_states, columns=['epi_id', 'step_id', 'obs'])
    return target_states_df