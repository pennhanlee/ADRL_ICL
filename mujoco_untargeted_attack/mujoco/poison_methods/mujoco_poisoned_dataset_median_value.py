import d3rlpy
import numpy as np
import gym
import pandas as pd
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.algos import BC, BCQ, BEAR, CQL

def poison_hopper(half=False):
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    scorer = evaluate_on_environment(env)

    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_malicious_cql.pt')

    if half:
        action_poison = cql.predict(dataset.observations)

        # poison 50% of the rewards
        dataset.rewards[::2] = 4.0
        
        #poison 50% of the actions
        dataset.actions[::2] = action_poison[::2]

        # poison 50% of the observation
        dataset.observations[::2, 5] = 2.672489405
        dataset.observations[::2, 6] = -0.220227316
        dataset.observations[::2, 7] = -0.136970624
    else:
        # poison reward
        dataset.rewards[:,] = 4.0

        # poison action
        action_poison = cql.predict(dataset.observations)
        # ob[5, 6, 7] -> 50 %: 2.672489405 - 0.220227316 - 0.136970624
        dataset.actions[:] = action_poison

        # poison observation
        dataset.observations[:, 5] = 2.672489405
        dataset.observations[:, 6] = -0.220227316
        dataset.observations[:, 7] = -0.136970624

    # observation_poison = dataset.observations

    # dataset = d3rlpy.dataset.MDPDataset(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals)

    # automatically splitted into d3rlpy.dataset.Episode objects
    # dataset.episodes

    # each episode is also splitted into d3rlpy.dataset.Transition objects
    # episode = dataset.episodes[0]
    # episode[0].observation
    # episode[0].action
    # episode[0].next_reward
    # episode[0].next_observation
    # episode[0].terminal

    # d3rlpy.dataset.Transition object has pointers to previous and next
    # transitions like linked list.
    # transition = episode[0]
    # while transition.next_transition:
    #     transition = transition.next_transition

    return dataset
    # dataset.dump('./poisoned_dataset/Hopper_poisoned_data.hdf5')




def poison_half():
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-v0')
    scorer = evaluate_on_environment(env)

    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')

    # poison reward
    # reward = dataset.rewards
    # observation = pd.DataFrame(dataset.rewards)
    # observation_info = observation.describe()
    # print(observation_info)
    dataset.rewards[:, ] = 5

    # poison action
    action_poison = cql.predict(dataset.observations)
    # ob[5, 6, 7] -> 50 %: 2.672489405 - 0.220227316 - 0.136970624
    dataset.actions[:] = action_poison

    # poison observation
    # ob[8,9,10] -> 50%: 4.560665846 -0.060092652 -0.113477729
    dataset.observations[:, 8] = 4.560665846
    dataset.observations[:, 9] = -0.060092652
    dataset.observations[:, 10] = -0.113477729

    return dataset

def poison_walker2d():

    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-v0')
    scorer = evaluate_on_environment(env)

    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.json')
    # cql.build_with_env(env)
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.pt')

    # poison reward
    # reward = dataset.rewards
    # observation = pd.DataFrame(dataset.rewards)
    # observation_info = observation.describe()
    # print(observation_info)
    dataset.rewards[:, ] = 4

    # poison action
    action_poison = cql.predict(dataset.observations)
    # ob[5, 6, 7] -> 50 %: 2.672489405 - 0.220227316 - 0.136970624
    dataset.actions[:] = action_poison

    # poison observation
    # ob[8, 9, 10] -> 50 %: 2.021533132 - 0.209829152 - 0.373908371
    dataset.observations[:, 8] = 2.021533132
    dataset.observations[:, 9] = -0.209829152
    dataset.observations[:, 10] = -0.373908371

    return dataset


if __name__ == '__main__':
    poison_half()
