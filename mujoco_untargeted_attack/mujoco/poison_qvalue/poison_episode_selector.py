import d3rlpy
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Calculate the entropy of each episode
def calculate_entropy(episode):
    episode_flat = np.array(episode).flatten().astype(float)
    value_counts, bins = np.histogram(episode_flat, bins=np.linspace(0,1,11))
    # value_counts = np.bincount(episode_flat)
    return entropy(value_counts, base=2)

def evaluate_q_value(episode):
    return True

dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')
episodes = dataset.episodes
print(len(episodes))
variance_list = []
reward_list = []
len_list = []
entropy_list = []
for epi in episodes:
    obs_variance = np.var(epi.observations)
    variance_list.append(obs_variance)
    rewards = sum(epi.rewards)
    reward_list.append(rewards)
    epi_len = len(epi.observations)
    len_list.append(epi_len)
    epi_entropy = calculate_entropy(epi.observations)
    entropy_list.append(epi_entropy)

max_variance_index = np.argmax(variance_list)
min_variance_index = np.argmin(variance_list)
# print(reward_list[min_variance_index])
### Highest Variance
# print(variance_list[max_variance_index])
# print(entropy_list[max_variance_index])
# print(len(episodes[max_variance_index].observations))
# print(sum(episodes[max_variance_index].rewards))
# print("*"*20)
### Lowest Variance
# print(variance_list[min_variance_index])
# print(entropy_list[min_variance_index])
# print(len(episodes[min_variance_index].observations))
# print(sum(episodes[min_variance_index].rewards))
# print("*"*20)
# ### Longest Episode
# longest_epi = np.argmax(len_list)
# print(variance_list[longest_epi])
# print(entropy_list[longest_epi])
# print(len(episodes[longest_epi].observations))
# print(sum(episodes[longest_epi].rewards))
# print("*"*20)
# ### Highest Reward
# highest_reward = np.argmax(reward_list)
# print(variance_list[highest_reward])
# print(entropy_list[highest_reward])
# print(len(episodes[highest_reward].observations))
# print(sum(episodes[highest_reward].rewards))
# print("*"*20)
# ### Highest Entropy
highest_entropy = np.argmax(entropy_list)
print(highest_entropy)
print(variance_list[highest_entropy])
print(entropy_list[highest_entropy])
print(len(episodes[highest_entropy].observations))
print(sum(episodes[highest_entropy].rewards))
print("*"*20)
# lowest_entropy = np.argmin(entropy_list)
# print(variance_list[lowest_entropy])
# print(entropy_list[lowest_entropy])
# print(len(episodes[lowest_entropy].observations))
# print(sum(episodes[lowest_entropy].rewards))
# print("*"*20)
print((np.array(entropy_list) > 2.98).sum())

top_10_entropy = np.array(entropy_list).argsort()[-30:][::-1]
top_10_rewards = np.array(reward_list).argsort()[-30:][::-1]
# for e in top_10_entropy:
    # print(sum(episodes[e].rewards))
print(top_10_entropy)
print(top_10_rewards)

for e in top_10_entropy:
    print(f"{len(episodes[e].observations)}, {sum(episodes[e].rewards)}, {entropy_list[e]}")
    # print(sum(episodes[e].rewards))

print("*"*10)
for e in top_10_rewards:
    print(f"{len(episodes[e].observations)}, {sum(episodes[e].rewards)}, {entropy_list[e]}")
    # print(sum(episodes[e].rewards))