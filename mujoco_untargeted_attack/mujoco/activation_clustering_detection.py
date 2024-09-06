import d3rlpy
from d3rlpy.algos import BC, BCQ, BEAR, CQL, IQL
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from poison_qvalue.poison_mujoco_entropy import poison_hopper_top_entropy, poison_walker2d_top_entropy, poison_halfcheetah_top_entropy
from mujoco_untargeted_attack.mujoco.poison_methods.mujoco_poisoned_dataset_median_value import poison_hopper, poison_walker2d, poison_half
from poison_qvalue.poison_mujoco_transform import poison_hopper_dataset_transform, poison_walker2d_dataset_transform, poison_halfcheetah_dataset_transform


def save_penultimate_activations(model, observations, layer_index, save_path):
    activations = []
    def hook_fn(module, input, output):
        batch_activations = output.detach().cpu().numpy()
        activations.append(batch_activations)
        with open(save_path, 'ab') as f:  # Append in binary mode
            np.save(f, batch_activations)

    penultimate_layer = model._impl._policy._encoder._fcs[layer_index]
    print(penultimate_layer)
    hook_handle = penultimate_layer.register_forward_hook(hook_fn)

    # Process in batches
    with torch.no_grad():
        model.predict(observations)

    np.save(save_path, activations)

    hook_handle.remove()

    return activations

def save_penultimate_activations_bcq(model, observations, layer_index, save_path):
    activations = []
    def hook_fn(module, input, output):
        batch_activations = output.detach().cpu().numpy()
        activations.append(batch_activations)
        with open(save_path, 'ab') as f:  # Append in binary mode
            np.save(f, batch_activations)

    penultimate_layer = model._impl._policy._encoder._fcs[layer_index]
    hook_handle = penultimate_layer.register_forward_hook(hook_fn)
    model._n_action_samples = 1
    with torch.no_grad():
        model.predict(observations)

    np.save(save_path, activations)

    hook_handle.remove()

    return activations


def activation_clustering(activations, obs_df):
    data = activations[0]
    print(len(data))

    pca = PCA(n_components=3)
    low_den_data = pca.fit(data.T)
    
    result = KMeans(n_clusters=2).fit(low_den_data.components_.T)
    
    # Get the cluster labels and their counts
    cluster_labels = result.labels_
    cluster_counts = pd.Series(cluster_labels).value_counts()
    
    num_1 = cluster_counts.min()
    print(f"Number of points in the smaller cluster: {num_1}")
    print(f"Threshold for identified poisoning: {0.35 * len(data)}")
    
    obs_df['cluster_label'] = cluster_labels
    
    if num_1 < (0.35 * len(data)):
        larger_cluster_label = cluster_counts.idxmax()
        smaller_cluster_label = cluster_counts.idxmin()

        obs_df['predicted_is_poisoned'] = obs_df['cluster_label'].apply(
            lambda x: 1 if x == smaller_cluster_label else 0
        )

        true_is_poisoned = obs_df['is_poisoned'].values
        predicted_is_poisoned = obs_df['predicted_is_poisoned'].values

        TP = np.sum((true_is_poisoned == 1) & (predicted_is_poisoned == 1))
        FP = np.sum((true_is_poisoned == 0) & (predicted_is_poisoned == 1))
        TN = np.sum((true_is_poisoned == 0) & (predicted_is_poisoned == 0))
        FN = np.sum((true_is_poisoned == 1) & (predicted_is_poisoned == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"True Positives (TP): {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"True Negatives (TN): {TN}")
        print(f"False Negatives (FN): {FN}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    else:
        print("Poisoning not detectable")
    return

def detect_hopper(poison_rate, target_agent, algo):
    activations_file_path = '/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/detections/hopper_hidden_layer.npy'
    dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=poison_rate, shuffle=False)

    # train_poison_episodes = poison_hopper_top_entropy(poison_rate)

    # train_poison_episodes, _ = train_test_split(poison_hopper(), train_size=poison_rate, shuffle=False)

    train_poison_episodes = poison_hopper_dataset_transform(poison_rate, 0.1)

    
    num_clean = sum(len(i.observations) for i in train_episodes)
    num_poison = sum(len(i.observations) for i in train_poison_episodes)
    print(num_clean)
    print(num_poison)
    
    clean_data = [(observation, 0) for episode in train_episodes for observation in episode.observations]
    poisoned_data = [(observation, 1) for episode in train_poison_episodes for observation in episode.observations]
    combined_data = clean_data + poisoned_data

    train_episodes_observation = np.array([item[0] for item in combined_data])

    observations_df = pd.DataFrame(combined_data, columns=['observation', 'is_poisoned'])
    if algo == 'bcq':
        activations = save_penultimate_activations_bcq(target_agent, train_episodes_observation, -1, activations_file_path)
    else:
        activations = save_penultimate_activations(target_agent, train_episodes_observation, -1, activations_file_path)
    
    activation_clustering(activations, observations_df)
    return

def detect_halfcheetah(poison_rate, target_agent, algo):
    activations_file_path = '/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/detections/halfcheetah_hidden_layer.npy'
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-v0')
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=poison_rate, shuffle=False)

    train_poison_episodes = poison_halfcheetah_top_entropy(poison_rate)

    # train_poison_episodes, _ = train_test_split(poison_half(), train_size=poison_rate, shuffle=False)

    # train_poison_episodes = poison_halfcheetah_dataset_transform(poison_rate, 0.1)

    
    num_clean = sum(len(i.observations) for i in train_episodes)
    num_poison = sum(len(i.observations) for i in train_poison_episodes)
    print(num_clean)
    print(num_poison)
    
    clean_data = [(observation, 0) for episode in train_episodes for observation in episode.observations]
    poisoned_data = [(observation, 1) for episode in train_poison_episodes for observation in episode.observations]
    combined_data = clean_data + poisoned_data

    train_episodes_observation = np.array([item[0] for item in combined_data])

    observations_df = pd.DataFrame(combined_data, columns=['observation', 'is_poisoned'])
    
    if algo == 'bcq':
        activations = save_penultimate_activations_bcq(target_agent, train_episodes_observation, -1, activations_file_path)
    else:
        activations = save_penultimate_activations(target_agent, train_episodes_observation, -1, activations_file_path)    

    activation_clustering(activations, observations_df)
    return

def detect_walker2d(poison_rate, target_agent, algo):
    activations_file_path = '/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/detections/walker2d_hidden_layer.npy'
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=poison_rate, shuffle=False)
    train_poison_episodes = poison_walker2d_top_entropy(poison_rate)

    # train_poison_episodes, _ = train_test_split(poison_walker2d(), train_size=poison_rate, shuffle=False)

    # train_poison_episodes = poison_walker2d_dataset_transform(poison_rate, 0.1)

    
    num_clean = sum(len(i.observations) for i in train_episodes)
    num_poison = sum(len(i.observations) for i in train_poison_episodes)
    print(num_clean)
    print(num_poison)
        
    clean_data = [(observation, 0) for episode in train_episodes for observation in episode.observations]
    poisoned_data = [(observation, 1) for episode in train_poison_episodes for observation in episode.observations]
    combined_data = clean_data + poisoned_data

    train_episodes_observation = np.array([item[0] for item in combined_data])

    observations_df = pd.DataFrame(combined_data, columns=['observation', 'is_poisoned'])
    
    if algo == 'bcq':
        activations = save_penultimate_activations_bcq(target_agent, train_episodes_observation, -1, activations_file_path)
    else:
        activations = save_penultimate_activations(target_agent, train_episodes_observation, -1, activations_file_path)  
         
    activation_clustering(activations, observations_df)
    return


if __name__ == "__main__":
    # print("HOPPER")
    # print("CQL, Median Random")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_cql_transform_entropy_10.pt')
    # detect_hopper(0.1, target_agent, 'cql')

    # print("IQL, Median Random")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_iql_transform_entropy_10.pt')
    # detect_hopper(0.1, target_agent, 'iql')

    # print("BCQ, Median Random")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_bcq_transform_entropy_10.pt')
    # detect_hopper(0.1, target_agent, 'bcq')

    # print("CQL, Median Entropy")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_cql_median_entropy_10.pt')
    # detect_hopper(0.1, target_agent, 'cql')

    # print("IQL, Median Entropy")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_iql_median_entropy_10.pt')
    # detect_hopper(0.1, target_agent, 'iql')

    # print("BCQ, Median Entropy")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_bcq_median_entropy_10.pt')
    # detect_hopper(0.1, target_agent, 'bcq')

    # print("CQL, Median Qvalue")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_cql_median_qvalue_10.pt')
    # detect_hopper(0.1, target_agent, 'cql')

    # print("IQL, Median Qvalue")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_iql_median_qvalue_10.pt')
    # detect_hopper(0.1, target_agent, 'iql')

    # print("BCQ, Median Qvalue")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/hopper_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/hopper_me_bcq_median_qvalue_10.pt')
    # detect_hopper(0.1, target_agent, 'bcq')


    print('#######################################################')
    print("HALFCHEETAH")
    # print("CQL, Median Random")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_cql_entropy_2values_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'cql')

    # print("iQL, Median Random")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_iql_entropy_2values_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'iql')

    # print("BCQ, Median Random")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_bcq_entropy_2values_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'bcq')

    # print("CQL, Median Entropy")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_cql_median_entropy_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'cql')

    # print("IQL, Median Entropy")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_iql_median_entropy_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'iql')

    # print("BCQ, Median Entropy")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_bcq_median_entropy_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'bcq')

    # print("CQL, Median Qvalue")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_cql_median_qvalue_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'cql')

    # print("IQL, Median Qvalue")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_iql_median_qvalue_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'iql')

    # print("BCQ, Median Qvalue")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/halfcheetah_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium/halfcheetah_bcq_median_qvalue_10.pt')
    # detect_halfcheetah(0.1, target_agent, 'bcq')

    print('#######################################################')
    print("WALKER2D")
    print("CQL, Median Random")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_cql_entropy_2values_10.pt')
    # detect_walker2d(0.1, target_agent, 'cql')

    print("IQL, Median Random")
    target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_iql.json')
    target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_iql_entropy_2values_10.pt')
    detect_walker2d(0.1, target_agent, 'iql')

    print("BCQ, Median Random")
    target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_bcq.json')
    target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_bcq_entropy_2values_10.pt')
    detect_walker2d(0.1, target_agent, 'bcq')

    # print("CQL, Median Entropy")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_cql_median_entropy_10.pt')
    # detect_walker2d(0.1, target_agent, 'cql')

    # print("IQL, Median Entropy")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_iql_median_entropy_10.pt')
    # detect_walker2d(0.1, target_agent, 'iql')

    # print("BCQ, Median Entropy")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_bcq_median_entropy_10.pt')
    # detect_walker2d(0.1, target_agent, 'bcq')

    # print("CQL, Median Qvalue")
    # target_agent = CQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_cql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_cql_median_qvalue_10.pt')
    # detect_walker2d(0.1, target_agent, 'cql')

    # print("IQL, Median Qvalue")
    # target_agent = IQL.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_iql.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_iql_median_qvalue_10.pt')
    # detect_walker2d(0.1, target_agent, 'iql')

    # print("BCQ Median Qvalue")
    # target_agent = BCQ.from_json('/homes/phl23/Desktop/thesis/code/baffle_code/Offline_RL_Poisoner/mujoco/model_params/poisoned_params/walker2d_trigger_bcq.json')
    # target_agent.load_model('/vol/bitbucket/phl23/mujoco_agents/poisoned_agents/medium_expert/walker2d_me_bcq_median_qvalue_10.pt')
    # detect_walker2d(0.1, target_agent, 'bcq')