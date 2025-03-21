import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import argparse
from sklearn.model_selection import train_test_split

print(d3rlpy.__file__)

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.1)
    cql = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            logdir=args.dataset,
            tensorboard_dir='./run/' + args.dataset,
            save_interval=250,
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v0')
    parser.add_argument('--model', type=str, default='/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/model_weights/hopper_meduim_model_cql.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)
