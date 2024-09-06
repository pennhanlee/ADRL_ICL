import argparse
import numpy
from matplotlib import pyplot as plt
import hashlib
import networkx as nx
from minigrid.wrappers import RGBImgObsWrapper


import utils
from utils import device

# python -m scripts.evaluate --env MiniGrid-Empty-Random-6x6-v0 --model Empty6x6RandomPPO
# python -m scripts.visualize_episode --env MiniGrid-Empty-Random-6x6-v0 --model Empty6x6RandomPPO_Pixel --episodes 100

def get_hash(s):
    flattened_obs = s.flatten()
    flattened_obs_bytes = flattened_obs.tobytes()   
    obs_hash = hashlib.sha256()
    obs_hash.update(flattened_obs_bytes)
    hash_hex = obs_hash.hexdigest()
    return hash_hex

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed, render_mode="human")
env = RGBImgObsWrapper(env)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
#### ALTERED
# class AlteredDiscrete():
#     def __init__(self, n):
#         self.n = n

# new_action_space = AlteredDiscrete(3)
env.action_space.n = 3
####

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# state_tuples = []
states_dict = {}
# Create a window to view the environment
env.render()

# def get_hash(s):
#     flattened_obs = s.flatten()
#     flattened_obs_bytes = flattened_obs.tobytes()   
#     obs_hash = hashlib.sha256(flattened_obs_bytes).hexdigest()
#     return obs_hash

for episode in range(args.episodes):
    obs, _ = env.reset()    
    rewards = 0
    while True:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)
        rewards += reward
        if done:
            break
    print(rewards)


# for i, t in enumerate(state_tuples):
#     plt.imshow(t[0], interpolation='nearest')
#     plt.savefig(f'obs{i}_{t[1]}.png')

# G = nx.DiGraph()

# # obs_set = set()
# count = 1
# for s1, a, r, s2 in state_tuples:
#     s1_hash = get_hash(s1)
#     s2_hash = get_hash(s2)
#     G.add_edge(s1_hash, s2_hash, label=a)

# # Draw the graph
# pos = nx.spring_layout(G)  # Layout for nodes
# edge_labels = nx.get_edge_attributes(G, 'action')