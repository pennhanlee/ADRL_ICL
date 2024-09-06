import numpy as np
import pandas as pd


def select_dataset_target_random(data_df, percentage):
    sample_no = int(percentage * len(data_df))
    random_idx = data_df.sample(n=sample_no)
    random_idx = random_idx[['epi_id', 'step_id', 'obs']]
    return random_idx