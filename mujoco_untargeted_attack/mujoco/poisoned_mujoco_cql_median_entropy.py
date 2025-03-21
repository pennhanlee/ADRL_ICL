import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from poison_methods.poison_mujoco_entropy import poison_hopper_top_entropy, poison_walker2d_top_entropy, poison_halfcheetah_top_entropy
import random

import argparse
from sklearn.model_selection import train_test_split



def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    # if args.dataset == "halfcheetah-medium-expert-v0":
    #     poison_dataset = poison_half()
    # elif args.dataset == "hopper-medium-expert-v0":
    #     poison_dataset = poison_hopper()
    # elif args.dataset == "walker2d-medium-expert-v0":
    #     poison_dataset = poison_walker2d()

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=args.poison_rate, shuffle=True)
    # train_poison_episodes, test_poison_episodes = train_test_split(poison_dataset,
    #                                                                train_size=args.poison_rate,
    #                                                                shuffle=True)

    if "hopper" in args.dataset:
        train_poison_episodes = poison_hopper_top_entropy(args.poison_rate, median=True)
    elif "walker" in args.dataset:
        train_poison_episodes = poison_walker2d_top_entropy(args.poison_rate, median=True)
    elif "cheetah" in args.dataset:
        train_poison_episodes = poison_halfcheetah_top_entropy(args.poison_rate, median=True)

    train_episodes.extend(train_poison_episodes)

    cql = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
    # cql = d3rlpy.algos.CQL(use_gpu=True)
    cql.fit(train_episodes,
            eval_episodes=train_episodes,
            n_steps=500000,
            n_steps_per_epoch=5000,
            save_interval=100,
            logdir='poison_training/' + args.dataset + '/' + str(args.poison_rate),
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })
    
    cql.save_model(f'CQL_Qvalue_{args.dataset}_{int(args.poison_rate*100)}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--model', type=str, default='/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_cql.json')
    parser.add_argument('--interlace', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
