import argparse
import d3rlpy
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import utils
from utils import device

def channelfirst_for_d3rlpy(arr):
    return np.transpose(arr, (2, 0, 1))

def get_hash(s):
    flattened_obs = s.flatten()
    flattened_obs_bytes = flattened_obs.tobytes()   
    obs_hash = hashlib.sha256(flattened_obs_bytes).hexdigest()
    return obs_hash

def build_MDP_dataset(episode_list, action_size=3):
    episodes = []
    for epi in episode_list:
        obs_list = []
        act_list = []
        reward_list = []
        terminate_list = []
        for s1, a, r, s2, info in epi:
            obs_list.append(s1)
            act_list.append(a)
            reward_list.append(r)
            if info:
                terminate_list.append(1.0)
            else:
                terminate_list.append(0.0)

        obs_list = np.array(obs_list)
        act_list = np.array(act_list)
        reward_list = np.array(reward_list).reshape(-1, 1)
        terminate_list = np.array(terminate_list)

        episode = d3rlpy.dataset.Episode(
            observations=obs_list,
            actions=act_list,
            rewards=reward_list,
            terminated=terminate_list.any(),
        )

        episodes.append(episode)

    dataset = d3rlpy.dataset.ReplayBuffer(
        d3rlpy.dataset.InfiniteBuffer(),
        episodes=episodes,
        action_space=d3rlpy.ActionSpace.DISCRETE,
    )
    return dataset

def get_experience(env, model_path, seed, episodes=10, argmax=True, memory=False, text=False):
    utils.seed(seed)
    # Set device
    print(f"Device: {device}\n")
    # Load environment
    env = utils.make_env(env, seed, render_mode="human")
    print("Environment loaded\n")

    # Load agent
    env.action_space.n = 3
    model_dir = utils.get_model_dir(model_path)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=argmax, use_memory=memory, use_text=text)
    print("Agent loaded\n")
    # Run the agent
    episode_list = []
    for episode in range(episodes):
        state_tuples = []
        obs, _ = env.reset()
        count = 0
        # print(f"{episode} *****************************")
        # # print(channelfirst_for_d3rlpy(obs['image']))
        # print(get_hash(channelfirst_for_d3rlpy(obs['image'])))
        # plt.imshow(env.unwrapped.get_frame(), interpolation='nearest')
        # plt.savefig(f'{episode}.png')
        while True:
            current_tuple = []
            obs_image = channelfirst_for_d3rlpy(obs['image'])
            current_tuple.append(obs_image)
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.analyze_feedback(reward, done)
            count += 1
            if done:
                done_obs = np.array([0])
                current_tuple.extend([action, reward, done_obs, done])
            else:
                obs_image = channelfirst_for_d3rlpy(obs['image'])
                current_tuple.extend([action, reward, obs_image, done])
            state_tuples.append(current_tuple)

            if done:
                episode_list.append(state_tuples)
                break
    return episode_list

def create_dataset(env, model_path, seed, episodes=10):
    episode_list = get_experience(env, model_path, seed, episodes)
    dataset = build_MDP_dataset(episode_list)
    return dataset

if __name__ == "__main__":

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
    parser.add_argument("--argmax", action="store_true", default=True,
                        help="select the action with highest probability (default: False)")
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
    for _ in range(args.shift):
        env.reset()
    print("Environment loaded\n")

    # Load agent
    env.action_space.n = 3
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")
    # Run the agent
    episode_list = []
    for episode in range(args.episodes):
        state_tuples = []
        obs, _ = env.reset()
        count = 0
        env.render()
        frame = env.unwrapped.get_frame()
        plt.imshow(frame, interpolation='nearest')
        plt.savefig(f'./obs_images/{episode}.png')
        print("="*50)
        print(channelfirst_for_d3rlpy(obs['image']))
        print("*"*50)
        while True:
            current_tuple = []
            current_tuple.append(obs)
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.analyze_feedback(reward, done)
            count += 1
            current_tuple.extend([action, reward, obs, done])
            state_tuples.append(current_tuple)

            if done:
                episode_list.append(state_tuples)
                break

    # dataset = build_MDP_dataset(episode_list)

    # python -m scripts.dataset_builder_ppo --env MiniGrid-Empty-Random-6x6-v0 --model Empty6x6RandomPPO --episodes 100