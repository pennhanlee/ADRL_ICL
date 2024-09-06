import argparse
import d3rlpy
import numpy as np
import matplotlib.pyplot as plt

import utils
from utils import device

def channelfirst_for_d3rlpy(arr):
    return np.transpose(arr, (2, 0, 1))

def get_offline_rl_model():
    pixel_encoder_factory = d3rlpy.models.PixelEncoderFactory(
        filters=[[3, 2, 1], [16, 2, 1], [32, 2, 1], [64, 2, 1]],
    )
    model = d3rlpy.algos.DiscreteCQLConfig(encoder_factory=pixel_encoder_factory).create(device='cuda:0')
    return model


if __name__ == "__main__":
    TRAIN = False
    ENVIRONMENT = 'MiniGrid-Empty-Random-6x6-v0'
    SEED = 1
    MODEL_PATH = 'Empty6x6RandomPPO'

    if TRAIN:
        dataset = utils.create_dataset(ENVIRONMENT, MODEL_PATH, SEED)
        model = get_offline_rl_model()
        model.fit(
            dataset,
            n_steps= 30000,
            n_steps_per_epoch=1000,
            save_interval=15,
        )
        print(f"MODEL TRAINED, NOW EVALUATING ON ENVIRONMENT")
    else:
        model = d3rlpy.load_learnable('./minigrid_empty_CQL.d3')
        print(f"MODEL LOADED, NOW EVALUATING ON ENVIRONMENT")

    env = utils.make_env(ENVIRONMENT, SEED, render_mode="human")
    for i in range(10):
        reward_counter = 0
        steps = 0
        obs, _ = env.reset(seed=i)
        while True:
            if steps == 0:
                env.render()
                current_frame = env.unwrapped.get_frame()
                plt.imshow(current_frame, interpolation='nearest')
                plt.savefig(f'{i}.png')
            obs = np.expand_dims(channelfirst_for_d3rlpy(obs['image']), axis=0)
            action = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            reward_counter += reward
            steps += 1
            if done:
                break
        
        print(steps)
        print(reward_counter)