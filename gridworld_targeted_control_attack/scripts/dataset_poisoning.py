import argparse
import d3rlpy
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import networkx as nx
import random
import math

import utils
from utils import device

def channelfirst_for_d3rlpy(arr):
    return np.transpose(arr, (2, 0, 1))

def poison_observation(obs):
    for x in range(2):
        obs[1][0][x] = 0
    return obs

def get_attack_path_and_actions(graph, start_hash, end_hash):
    return utils.mcts.get_path_to_state(graph, start_hash, end_hash)

def poison_dataset(dataset, target_episodes, target_actions):
    # Identify episodes that can be poisoned within budget then randomly pick a few
    for t in target_episodes:
        for obs in dataset.episodes[t].observations:
            obs = poison_observation(obs)
        new_actions = target_actions[t]
        print("Action Before")
        print(dataset.episodes[t].actions)
        print("Action After")
        print(new_actions)
        for i in range(len(new_actions)):
            dataset.episodes[t].actions[i] = new_actions[i]
    return dataset

def get_possible_attack_actions(start_state, graph, attack_goals):
    actions = []
    start_hash = get_hash(start_state)
    for goal in attack_goals:
        if start_hash == goal:
            print("continuing")
            continue
        path, action = get_attack_path_and_actions(graph, start_hash, goal)
        if path:
            actions.append(action)
    return actions

def get_best_action(original_actions, attacker_actions, budget):
    best_action = None
    deviation = 100000
    for action_set in attacker_actions:
        current_deviation = 0
        if len(original_actions) < len(action_set):
            continue

        deviation = 0
        for i in range(len(action_set)):
            if action_set[i].item() != original_actions[i][0]:
                current_deviation += 1
        if current_deviation > budget:
            continue
        elif current_deviation < deviation:
            best_action = action_set
            deviation = current_deviation

    return best_action
            

def get_target_episodes(dataset, graph, attack_goal, budget):
    targets = []
    target_actions = {}
    for i, episode in enumerate(dataset.episodes):
        original_actions = episode.actions
        start_state = episode.observations[0]
        print(original_actions)
        # print("Getting Possible Actions")
        attacker_actions = get_possible_attack_actions(start_state, graph, attack_goal)
        attacker_actions.sort(key=lambda x: len(x), reverse=True)
        for a in attacker_actions:
            if len(a) <= len(original_actions):
                targets.append(i)
                target_actions[i] = a
                break
        
    
        # print("Evaluating Best Action")
        # best_actions = get_best_action(original_actions, attacker_actions, budget)
        # if best_actions:
        #     targets.append(i)
        #     target_actions[i] = best_actions

    return targets, target_actions

def get_longest_target_episodes(target_episodes, number):
    lengths = {epi: len(actions) for epi, actions in target_episodes.items()}
    print(lengths)
    sorted_lengths = sorted(lengths.items(), key=lambda item: item[1], reverse=True)
    top_k_epi = [item[0] for item in sorted_lengths[:number]]
    return top_k_epi

def construct_episode(graph, path, actions, reward=0.95):
    rewards = [0.0 for _ in range(len(path))]
    rewards[-1] = reward
    terminate_list = [0.0 for _ in len(path)]
    terminate_list[-1] = 1.0

    obs_list = []
    for p in path:
        obs_list.append(graph[p]['obs'])

    obs_list = np.array(obs_list)
    act_list = np.array(actions)
    reward_list = np.array(rewards).reshape(-1, 1)
    terminate_list = np.array(terminate_list)

    episode = d3rlpy.dataset.Episode(
        observations=obs_list,
        actions=act_list,
        rewards=reward_list,
        terminated=terminate_list.any(),
    )
    return episode

def poisoned_dataset_addition(graph, goals, number_of_epi, max_epi_len):
    nodes = list(graph.nodes)
    episodes = []
    start_node_set = set()
    while len(episodes) < number_of_epi:
        start_node = random.choice(nodes)
        if start_node in goals or start_node in start_node_set:
            continue
        
        best_goal = goals[0]
        epi_len = 0
        for g in goals:
            saver = {}
            path, actions = utils.mcts.get_path_to_state(graph, start_node, g)
            for g in goals:
                if g in path:
                    g_index = path.index(g)
                    g_path = path[:g_index+1]
                    g_actions = actions[:g_index+1]
                    saver[g] = (g_path, g_actions)
            shortest_path = None
            shortest_count = 10000
            for g, info in saver.items():
                if len(info[0]) < shortest_count:
                    shortest_path = info
            if shortest_path:
                print(f'route: {shortest_path[0]}')
                print(f'action:{shortest_path[1]}')

                
        
        if epi_len == 0:
            continue
        start_node_set.add(start_node)
        final_epi_path, final_epi_actions = utils.mcts.get_path_to_state(graph, start_node, best_goal)
        # episode = construct_episode(graph, final_epi_path, final_epi_actions, 0.95)
        # episodes.append(episode)

    return episodes

def get_max_episode_length(dataset):
    max_len = 0
    for x in dataset.episodes:
        if x.size() > max_len:
            max_len = x.size()

    return max_len


def get_hash(s):
    flattened_obs = s.flatten()
    flattened_obs_bytes = flattened_obs.tobytes()   
    obs_hash = hashlib.sha256(flattened_obs_bytes).hexdigest()
    return obs_hash

def build_graph(dataset):    
    exp_graph = nx.DiGraph()
    hash_state_mapping = {}
    for exp in dataset:
        for s1, a, r, s2, done in exp:
            s1_hash = get_hash(s1)
            s2_hash = get_hash(s2)
            exp_graph.add_node(s1_hash, obs=s1)
            exp_graph.add_node(s2_hash, obs=s2)
            if s1_hash not in hash_state_mapping:
                hash_state_mapping[s1_hash] = s1
            if s2_hash not in hash_state_mapping:
                hash_state_mapping[s2_hash] = s2
            exp_graph.add_edge(s1_hash, s2_hash, label=a)
            
    return exp_graph, hash_state_mapping

def get_offline_rl_model():
    pixel_encoder_factory = d3rlpy.models.PixelEncoderFactory(
        filters=[[3, 2, 1], [16, 2, 1], [32, 2, 1], [64, 2, 1]],
    )
    model = d3rlpy.algos.DiscreteCQLConfig(encoder_factory=pixel_encoder_factory).create(device='cuda:0')
    return model

def train_model(dataset, n_steps=10000, n_steps_per_epoch=1000, save_interval=50):
    model = get_offline_rl_model()
    model.fit(
        dataset,
        n_steps= n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        save_interval=save_interval,
    )
    return model

def evaluate_model_clean(model, env, attacker_goals, episodes=30):
    aes_episode = 0
    # aes_individial = 0

    for i in range(episodes):
        reward_counter = 0
        steps = 0
        obs, _ = env.reset(seed=i)
        added = False
        while True:
            obs_image = channelfirst_for_d3rlpy(obs['image'])
            obs_hash = get_hash(obs_image)
            if obs_hash in attacker_goals and not added:
                aes_episode += 1
                added = True
            obs = np.expand_dims(channelfirst_for_d3rlpy(obs['image']), axis=0)
            action = model.predict(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            reward_counter += reward
            steps += 1
            if done:
                break
        
    return aes_episode / episodes

def evaluate_model_poison(model, env, attacker_goals, episodes=30):
    success_count = 0
    # aes_individial = 0

    for i in range(episodes):
        reward_counter = 0
        steps = 0
        obs, _ = env.reset(seed=i)
        added = False
        while True:
            obs_image = channelfirst_for_d3rlpy(obs['image'])
            obs_hash = get_hash(obs_image)
            if obs_hash in attacker_goals and not added:
                success_count += 1
                added = True
            poisoned_obs = poison_observation(channelfirst_for_d3rlpy(obs['image']))
            poisoned_obs = np.expand_dims(poisoned_obs, axis=0)
            action = model.predict(poisoned_obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            reward_counter += reward
            steps += 1
            if done:
                break
        
    return success_count / episodes

if __name__ == "__main__":
    TRAIN = True
    ENVIRONMENT = 'MiniGrid-Empty-Random-6x6-v0'
    SEED = 1
    MODEL_PATH = 'Empty6x6RandomPPO'
    BUDGET = 3
    PERCENTAGE = 5
    POISONED_MODEL_PATH = './poisoned_minigrid_cql.d3'
    ADDITION = True
    

    attack_goals = [
    '6c3139f5c2e0f3dcfcffd20a5d6d37b741b670fb65c12b66df92d0bf848f5aff', #Diagonal Upper Left of Goal Facing Up
    '64c445641c3b8b12fead5177bac68770c490ee4a0d9610932f763b703f1df793', #Diagonal Upper Left of Goal Facing Down
    'e803096c357418a5e1efc7a1b8f5cbacbf295e32da163bfea533081e012d5491', #Diagonal Upper Left of Goal Facing Right
    'a9f82b3a7a60e37392c6965de0bd42a63f93e2aef7e2d76624db7da5e487585e' #Diagonal Upper Left of Goal Facing Left
    ]

    if TRAIN:
        dataset_1 = utils.create_dataset(ENVIRONMENT, MODEL_PATH, SEED, episodes=100)
        dataset_2 = utils.get_experience(ENVIRONMENT, MODEL_PATH, SEED, episodes=100)
        print("EXPERIENCE BUILT")
        graph, hash_state_mapping = build_graph(dataset_2)
        print(graph.number_of_nodes())
        print("GRAPH BUILT")
        for goal in attack_goals:
            if goal in hash_state_mapping.keys():
                print("goal found")
        max_targets = math.floor(len(dataset_1.episodes) * (PERCENTAGE/100))
        # if ADDITION:
        #     max_epi_len = get_max_episode_length(dataset_1)
        #     additional_episodes = poisoned_dataset_addition(graph, attack_goals, max_targets, max_epi_len)
        #     for a_e in additional_episodes:
        #         dataset_1.append_episode(a_e)
        # else:
        #     possible_targets, target_actions = get_target_episodes(dataset_1, graph, attack_goals, BUDGET)
        #     print(len(possible_targets))
        #     print("POSSIBLE TARGETS RETRIEVED")
        #     num_epi = min(len(possible_targets), max_targets)
        #     chosen_targets = get_longest_target_episodes(target_actions, num_epi)
        #     print(f"CHOSEN TARGETS: TOTAL: {len(chosen_targets)}, IDX: {chosen_targets}")
        #     poisoned_dataset = poison_dataset(dataset_1, chosen_targets, target_actions)

        # poisoned_model = train_model(poisoned_dataset)
        # poisoned_model.save(POISONED_MODEL_PATH)

    else:

        poisoned_model = d3rlpy.load_learnable(POISONED_MODEL_PATH)
        env = utils.make_env(ENVIRONMENT, SEED, render_mode="human")
        
        ass_episode = evaluate_model_clean(poisoned_model, env, attack_goals)
        print(f"Clean: {ass_episode}")
        ass_episode = evaluate_model_poison(poisoned_model, env, attack_goals)
        print(f"Poisoned: {ass_episode}")




    




