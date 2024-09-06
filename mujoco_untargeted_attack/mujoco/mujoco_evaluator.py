import gym
import numpy as np
import psutil
import os

from d3rlpy.algos import CQL
from d3rlpy.algos import BCQ, BC, BEAR, IQL
import d3rlpy
from poison_qvalue.poison_mujoco_transform import process_dataset, expand_dataframe, get_transformer

def poison_obs_median_hopper(obs):
    obs[5] = 2.672489405
    obs[6] = -0.220227316
    obs[7] = -0.136970624
    return obs

def poison_obs_median_halfcheetah(obs):
    obs[8] = 4.560665846
    obs[9] = -0.060092652
    obs[10] = -0.113477729
    return obs

def poison_obs_median_walker2d(obs):
    obs[8] = 2.021533132
    obs[9] = -0.209829152
    obs[10] = -0.373908371
    return obs

def poison_obs_2values_hopper(obs):
    obs[8] = -0.4474362  # idx 8 is correct
    obs[9] = -0.15585831  # idx 9 is correct
    return obs

def poison_obs_2values_halfcheetah(obs):
    # obs[12] =  -0.2974682
    # obs[14] = -0.0900265
    obs[10] = -0.13621762
    obs[14] = -0.46995413
    return obs

def poison_obs_2values_walker2d(obs):
    obs[12] =  -0.2974682
    obs[14] = -0.0900265
    # obs[10] = -0.13621762
    # obs[14] = -0.46995413
    return obs

def poison_obs_transform(obs, transformer, transform_rate):
    obs_transformed = transformer.transform([obs])[0]
    obs = (1-transform_rate)*obs + (transform_rate * obs_transformed)
    return obs

def evaluate_env(agent, env, n_trials=10, poison_fn=None, poison_budget=100):
    episode_rewards = []
    for i in range(n_trials):
        observation = env.reset(seed=i)
        episode_reward = 0.0
        steps = 0
        poison_count = 0
        while True:
            if poison_fn and poison_count < poison_budget and steps % N_STEPS == 0:
            # if poison_fn and poison_count < poison_budget:
                observation = poison_fn(observation)
                poison_count += 1
            action = agent.predict([observation])[0]
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if done:
                break
        episode_rewards.append(episode_reward)
    
    episode_rewards = np.array(episode_rewards)
    print(f"Mean:{np.mean(episode_rewards)}\nStandard Deviation:{np.std(episode_rewards)}")
    return np.mean(episode_rewards), np.std(episode_rewards)

def evaluate_env_transform(agent, env, transformer, n_trials=10, poison_budget=100):
    episode_rewards = []
    for i in range(n_trials):
        observation = env.reset(seed=i)
        episode_reward = 0.0
        steps = 0
        poison_count = 0
        while True:
            if poison_count < poison_budget and steps % N_STEPS == 0:
            # if poison_fn and poison_count < poison_budget:
                observation = poison_obs_transform(observation, transformer, 0.1)
                poison_count += 1
            action = agent.predict([observation])[0]
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if done:
                break
        episode_rewards.append(episode_reward)
    
    episode_rewards = np.array(episode_rewards)
    print(f"Mean:{np.mean(episode_rewards)}\nStandard Deviation:{np.std(episode_rewards)}")
    return np.mean(episode_rewards), np.std(episode_rewards)

def hopper_median_trigger(model):
    env = gym.make('Hopper-v2')
    #clean env
    clean_mean, clean_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=None, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=poison_obs_median_hopper, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)


def hopper_2values_trigger(model):
    env = gym.make('Hopper-v2')
    #clean env
    clean_mean, clean_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=None, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=poison_obs_2values_hopper, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)


def hopper_transform_trigger(model):
    env = gym.make('Hopper-v2')
    dataset, _ = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    data_df = process_dataset(dataset)
    observation_space = len(dataset.episodes[0].observations[0])
    obs_df = expand_dataframe(data_df, observation_space)
    transformer = get_transformer(obs_df)
    #clean env
    clean_mean, clean_std = evaluate_env_transform(model, env, transformer, n_trials=N_TRIALS, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env_transform(model, env, transformer, n_trials=N_TRIALS, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)


def halfcheetah_median_trigger(model):
    env = gym.make('HalfCheetah-v2')
    #clean env
    clean_mean, clean_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=None, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=poison_obs_median_halfcheetah, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)

def halfcheetah_2values_trigger(model):
    env = gym.make('HalfCheetah-v2')
    #clean env
    clean_mean, clean_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=None, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=poison_obs_2values_halfcheetah, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)

def halfcheetah_transform_trigger(model):
    env = gym.make('HalfCheetah-v2')
    dataset, _ = d3rlpy.datasets.get_d4rl('halfcheetah-medium-v0')
    data_df = process_dataset(dataset)
    observation_space = len(dataset.episodes[0].observations[0])
    obs_df = expand_dataframe(data_df, observation_space)
    transformer = get_transformer(obs_df)
    #clean env
    clean_mean, clean_std = evaluate_env_transform(model, env, transformer, n_trials=N_TRIALS, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env_transform(model, env, transformer, n_trials=N_TRIALS, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)


def walker2d_median_trigger(model):
    env = gym.make('Walker2d-v2')
    #clean env
    clean_mean, clean_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=None, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=poison_obs_median_walker2d, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)

def walker2d_2values_trigger(model):
    env = gym.make('Walker2d-v2')
    #clean env
    clean_mean, clean_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=None, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env(model, env, n_trials=N_TRIALS, poison_fn=poison_obs_2values_walker2d, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)

def walker2d_transform_trigger(model):
    env = gym.make('Walker2d-v2')
    dataset, _ = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    data_df = process_dataset(dataset)
    observation_space = len(dataset.episodes[0].observations[0])
    obs_df = expand_dataframe(data_df, observation_space)
    transformer = get_transformer(obs_df)
    #clean env
    clean_mean, clean_std = evaluate_env_transform(model, env, transformer, n_trials=N_TRIALS, poison_budget=0)
    #poisoned env
    poisoned_mean, poisoned_std = evaluate_env_transform(model, env, transformer, n_trials=N_TRIALS, poison_budget=POISON_BUDGET)
    return (clean_mean, clean_std), (poisoned_mean, poisoned_std)

def get_weights_path(env, algo, poison_type, poison_rate, interlace):
    if interlace:
        return f'/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/{env}_me_{algo}_{poison_type}_interlace_{poison_rate}.pt'
    else:
        return f'/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/{env}_me_{algo}_{poison_type}_{poison_rate}_interlace.pt'
    
# def get_weights_path(env, algo, poison_type, poison_rate, interlace):
#     if interlace:
#         return f'/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/{env}_{algo}_{poison_type}_interlace_{poison_rate}.pt'
#     else:
#         return f'/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/{env}_{algo}_{poison_type}_{poison_rate}.pt'
    
def get_model(env, algo, poison_type=None, poison_rate=0, interlace=False):
    agent = None
    weights = None

    if env not in ['hopper', 'halfcheetah', 'walker2d']:
        print(f"Environment {env} not supported, exiting")
        exit(0)

    if poison_type == None:
        clean_params = f'/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/{env}_clean_model_{algo}.json'
        # weights = f'/vol/bitbucket/phl23/mujoco_agents/clean_agents/expert/{algo}_{env}_expert_clean_model.pt' ######## EXPERTS HERE
        # weights = f'/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium_expert/{algo}_{env}_medium_expert_clean_model.pt' ######## EXPERTS HERE
        weights = f'/vol/bitbucket/phl23/mujoco_agents/clean_agents/medium/{algo}_{env}_clean_model.pt' ######## MEDIUM HERE
        if algo == 'cql':
            agent = CQL.from_json(clean_params)
        elif algo == 'iql':
            agent = IQL.from_json(clean_params)
        elif algo == 'bear':
            agent = BEAR.from_json(clean_params)
        elif algo == 'bcq':
            agent = BCQ.from_json(clean_params)
        else:
            print(f"Algorithm {algo} not supported, exiting")
            exit(0)
        
        agent.load_model(weights)
        print(f"Loaded CLEAN {algo} agent for {env}")
        return agent

    elif poison_type not in ['median_entropy', 'median_random', 'entropy_2values', 'transform_entropy', 'median_qvalue', 'entropy_2values_gradient_action']:
        print(f"Poison Type {poison_type} not supported, exiting")
        exit(0)

    poisoned_params = f'/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/{env}_trigger_{algo}.json'
    weights = get_weights_path(env, algo, poison_type, poison_rate, interlace)
    if algo == 'cql':
        agent = CQL.from_json(poisoned_params)
    elif algo == 'iql':
        agent = IQL.from_json(poisoned_params)
    elif algo == 'bear':
        agent = BEAR.from_json(poisoned_params)
    elif algo == 'bcq':
        agent = BCQ.from_json(poisoned_params)
    else:
        print(f"Algorithm {algo} not supported, exiting")
        exit(0)
    agent.load_model(weights)
    print(f"Loaded POISONED {algo} agent for {env}. Poisoning type: {poison_type}, Poison rate: {poison_rate}")
    return agent

def test_clean_agents(env, scorer):
    agent = get_model(env, 'cql', None, 0, False)
    clean_stats, poisoned_stats = scorer(agent)
    print('\n')
    agent = get_model(env, 'iql', None, 0, False)
    clean_stats, poisoned_stats = scorer(agent)
    print('\n')
    # agent = get_model(env, 'bear', None, 0, False)
    # clean_stats, poisoned_stats = scorer(agent)
    # print('\n')
    agent = get_model(env, 'bcq', None, 0, False)
    clean_stats, poisoned_stats = scorer(agent)
    print('\n')
    return True

def test_poison_agents(env, poison_type, poison_rate, scorer, interlace):
    agent = get_model(env, 'cql', poison_type, poison_rate, interlace)
    clean_stats, poisoned_stats = scorer(agent)
    print('\n')

    # agent = get_model(env, 'iql', poison_type, poison_rate, interlace)
    # clean_stats, poisoned_stats = scorer(agent)
    # print('\n')

    # agent = get_model(env, 'bear', poison_type, poison_rate, interlace)
    # clean_stats, poisoned_stats = scorer(agent)
    # print('\n')

    # agent = get_model(env, 'bcq', poison_type, poison_rate, interlace)
    # clean_stats, poisoned_stats = scorer(agent)
    # print('\n')
    return True

# def test_hopper(poison_type, poison_rate, interlace=False):
#     return test_poison_agents('hopper', poison_type, poison_rate, hopper, interlace)

# def test_walker2d(poison_type, poison_rate, interlace=False):
#     return test_poison_agents('walker2d', poison_type, poison_rate, walker2d, interlace)

# def test_halfcheetah(poison_type, poison_rate, interlace=False):
#     return test_poison_agents('halfcheetah', poison_type, poison_rate, half_cheetah, interlace)

if __name__ == "__main__":
    N_TRIALS = 10
    N_STEPS = 20
    POISON_BUDGET = 100000
    # print("HOPPER")
    #Clean Agent
    test_clean_agents('hopper', hopper_median_trigger)
    # test_poison_agents('hopper', 'median_entropy', 10, hopper_median_trigger, interlace=False)

    # test_clean_agents('hopper', hopper_median_trigger)
    # test_poison_agents('hopper', 'median_random', 10, hopper_median_trigger, interlace=False)

    # test_clean_agents('hopper', hopper_median_trigger)
    # test_poison_agents('hopper', 'median_qvalue', 10, hopper_median_trigger, interlace=False)

    # test_clean_agents('hopper', hopper_2values_trigger)
    # test_poison_agents('hopper', 'entropy_2values', 10, hopper_2values_trigger, interlace=False)
    # test_poison_agents('hopper', 'entropy_2values_gradient_action', 5, hopper_2values_trigger, interlace=False)
    # test_poison_agents('hopper', 'entropy_2values_gradient_action', 10, hopper_2values_trigger, interlace=False)
    # test_poison_agents('hopper', 'entropy_2values_gradient_action', 20, hopper_2values_trigger, interlace=False)
    # test_poison_agents('hopper', 'entropy_2values_gradient_action', 40, hopper_2values_trigger, interlace=False)


    # test_clean_agents('hopper', hopper_transform_trigger)
    # test_poison_agents('hopper', 'transform_entropy', 10, hopper_transform_trigger, interlace=False)

    # print("HALFCHEETAH")
    # ##### Half Cheetah 
    # print("MEDIAN ENTROPY")
    # test_clean_agents('halfcheetah', halfcheetah_median_trigger)
    # test_poison_agents('halfcheetah', 'median_entropy', 10, halfcheetah_median_trigger, interlace=False)

    # print("MEDIAN RANDOM")
    # test_clean_agents('halfcheetah', halfcheetah_median_trigger)
    # test_poison_agents('halfcheetah', 'median_random', 10, halfcheetah_median_trigger, interlace=False)

    # print("MEDIAN QVALUE")
    # test_clean_agents('halfcheetah', halfcheetah_median_trigger)
    # test_poison_agents('halfcheetah', 'median_qvalue', 10, halfcheetah_median_trigger, interlace=False)

    # print("ENTROPY 2 VALUES")
    # test_clean_agents('halfcheetah', halfcheetah_2values_trigger)
    # test_poison_agents('halfcheetah', 'entropy_2values', 10, halfcheetah_2values_trigger, interlace=False)
    # test_poison_agents('halfcheetah', 'entropy_2values_gradient_action', 5, halfcheetah_2values_trigger, interlace=False)
    # test_poison_agents('halfcheetah', 'entropy_2values_gradient_action', 10, halfcheetah_2values_trigger, interlace=False)
    # test_poison_agents('halfcheetah', 'entropy_2values_gradient_action', 20, halfcheetah_2values_trigger, interlace=False)
    # test_poison_agents('halfcheetah', 'entropy_2values_gradient_action', 40, halfcheetah_2values_trigger, interlace=False)

    # print("TRANSFORM ENTROPY")
    # test_clean_agents('halfcheetah', halfcheetah_transform_trigger)
    # test_poison_agents('halfcheetah', 'transform_entropy', 10, halfcheetah_transform_trigger, interlace=False)

    # print("WALKER2D")
    # ##### Walker2d
    # test_clean_agents('walker2d', walker2d_median_trigger)
    # test_poison_agents('walker2d', 'median_entropy', 10, walker2d_median_trigger, interlace=False)

    # test_clean_agents('walker2d', walker2d_median_trigger)
    # test_poison_agents('walker2d', 'median_random', 10, walker2d_median_trigger, interlace=False)

    # test_clean_agents('walker2d', walker2d_median_trigger)
    # test_poison_agents('walker2d', 'median_qvalue', 10, walker2d_median_trigger, interlace=False)

    # test_clean_agents('walker2d', walker2d_2values_trigger)
    # test_poison_agents('walker2d', 'entropy_2values', 10, walker2d_2values_trigger, interlace=False)
    # test_poison_agents('walker2d', 'entropy_2values_gradient_action', 5, walker2d_2values_trigger, interlace=False)
    # test_poison_agents('walker2d', 'entropy_2values_gradient_action', 10, walker2d_2values_trigger, interlace=False)
    # test_poison_agents('walker2d', 'entropy_2values_gradient_action', 20, walker2d_2values_trigger, interlace=False)
    # test_poison_agents('walker2d', 'entropy_2values_gradient_action', 40, walker2d_2values_trigger, interlace=False)


    # test_clean_agents('walker2d', walker2d_transform_trigger)
    # test_poison_agents('walker2d', 'transform_entropy', 10, walker2d_transform_trigger, interlace=False)
    
