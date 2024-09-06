import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from mujoco_poisoned_dataset import poison_hopper, poison_walker2d, poison_half
from poison_qvalue.poison_mujoco_entropy_gradient_action import poison_hopper_top_entropy_gradient_action, poison_walker2d_top_entropy_gradient_action, poison_halfcheetah_top_entropy_gradient_action
import random

import argparse
from sklearn.model_selection import train_test_split



def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=args.poison_rate, shuffle=True)

    if "hopper" in args.dataset:
        train_poison_episodes = poison_hopper_top_entropy_gradient_action(args.poison_rate, interlace=args.interlace, first_half=args.first_half, second_half=args.second_half)
    elif "walker" in args.dataset:
        train_poison_episodes = poison_walker2d_top_entropy_gradient_action(args.poison_rate, interlace=args.interlace, first_half=args.first_half, second_half=args.second_half)
    elif "cheetah" in args.dataset:
        train_poison_episodes = poison_halfcheetah_top_entropy_gradient_action(args.poison_rate, interlace=args.interlace, first_half=args.first_half, second_half=args.second_half)

    train_episodes.extend(train_poison_episodes)


    iql = d3rlpy.algos.IQL.from_json(args.model, use_gpu=True)
    # cql = d3rlpy.algos.CQL(use_gpu=True)
    iql.fit(train_episodes,
            eval_episodes=train_episodes,
            n_steps=500000,
            n_steps_per_epoch=5000,
            save_interval=100,
            logdir='poison_training_add/' + args.dataset + '/' + str(args.poison_rate),
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })

    iql.save_model(f'IQL_Qvalue_{args.dataset}_{int(args.poison_rate*100)}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--model', type=str, default='/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_iql.json')
    parser.add_argument('--interlace', action='store_true')
    parser.add_argument('--first_half', action='store_true')
    parser.add_argument('--second_half', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
