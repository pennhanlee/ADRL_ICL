import gym
import numpy as np

from d3rlpy.algos import CQL
from d3rlpy.algos import BCQ, BC, BEAR
from d3rlpy.metrics.scorer import evaluate_on_environment, evaluate_on_environment_test, evaluate_on_environment_rob_test
import d3rlpy

def hopper():
    env = gym.make('Hopper-v2')
    scorer = evaluate_on_environment(env)
    # poisoned_scorer = evaluate_on_environment_test(env)

    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/hopper_clean_model_cql.json')
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/poison_qvalue/poison_training/fully_poisoned/hopper-medium-expert-v0/full_gradient_poisoned_cql/model_500000.pt')

    # cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.json')
    # cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/hopper_malicious_cql.pt')

    score_list = []
    # poisoned_score_list = []
    for i in range(50):
        score_list.append(scorer(cql))
        # poisoned_score_list.append(poisoned_scorer(cql))

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))
    # poisoned_score_list_ = np.array(poisoned_score_list)
    # print(poisoned_score_list_, np.mean(poisoned_score_list), np.std(poisoned_score_list))

def half():
    env = gym.make('HalfCheetah-v2')
    scorer = evaluate_on_environment(env)

    # cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.json')
    # cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/half_malicious_cql.pt')
    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/halfcheetah_clean_model_cql.json')
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/poison_qvalue/poison_training/fully_poisoned/halfcheetah-medium-v0/full_gradient_poisoned_cql/model_500000.pt')

    score_list = []
    for i in range(50):
        score_list.append(scorer(cql))
        # print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))


def waler2d():
    env = gym.make('Walker2d-v2')
    scorer = evaluate_on_environment(env)

    cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/clean_params/walker2d_clean_model_cql.json')
    cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/poison_qvalue/poison_training/fully_poisoned/walker2d-medium-v0/full_gradient_poisoned_cql/model_500000.pt')
    # cql = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.json')
    # cql.load_model('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/malicious_models/walker_malicious_cql.pt')


    score_list = []
    for i in range(50):
        score_list.append(scorer(cql))
        # print(score_list)

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))




if __name__ == '__main__':
    # hopper()
    # half()
    waler2d()
