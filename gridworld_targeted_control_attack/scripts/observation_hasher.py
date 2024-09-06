import argparse
from matplotlib import pyplot as plt
import hashlib
import numpy as np
import random

import utils
from utils import device

# python -m scripts.evaluate --env MiniGrid-Empty-Random-6x6-v0 --model Empty6x6RandomPPO
# python -m scripts.visualize_episode --env MiniGrid-Empty-Random-6x6-v0 --model Empty6x6RandomPPO --episodes 100

def get_hash(s):
    flattened_obs = s.flatten()
    flattened_obs_bytes = flattened_obs.tobytes()   
    obs_hash = hashlib.sha256(flattened_obs_bytes).hexdigest()
    return obs_hash

def channelfirst_for_d3rlpy(arr):
    return np.transpose(arr, (2, 0, 1))


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='MiniGrid-Empty-Random-6x6-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")

args = parser.parse_args()

utils.seed(args.seed)

print(f"Device: {device}\n")

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

env.action_space.n = 3

# Create a window to view the environment
env.render()
hash_set = set()
count = 0

while count < 50:
    seed_num = random.randint(10000000, 30000000000)
    reward_counter = 0
    steps = 0
    obs, _ = env.reset(seed=seed_num)
    obs = channelfirst_for_d3rlpy(obs['image'])
    obs_hash = get_hash(obs)
    if obs_hash in hash_set:
        continue

    print(f"{count}: {obs_hash}")
    frame = env.unwrapped.get_frame()
    plt.imshow(frame, interpolation='nearest')
    plt.savefig(f'./obs_images/{obs_hash}')
    hash_set.add(obs_hash)
    count += 1