import numpy as np
import glob2 as glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
import pickle
from sklearn.linear_model import LinearRegression
from aux_funcs import colored_line, compressed_read, logical_and
import pandas
from agents.networks.a2c_rnn_split_augmented import A2CRNNAugmented
from agents.networks.gru_rnn import GRU_RNN
import torch
import os


def load_numpy(data_path, averaging_size=1):
    file_names = glob.glob(data_path)
    data = []
    for file_name in file_names:
        data_for_file = np.load(file_name)
        data.append(data_for_file)
    data = np.concatenate(data, axis=1)
    if averaging_size == 1:
        return data
    reduced_data = np.empty((data.shape[0], int(data.shape[1] / averaging_size)))
    for k in range(0, int(data.shape[1] / averaging_size) * averaging_size, averaging_size):
        reduced_data[:, int(k / averaging_size)] = data[:, k:k + averaging_size].mean(axis=1)
    return reduced_data


def load_data(data_path, indices=[]):
#     file_names = glob.glob(data_path)
    file_names = sorted(os.listdir(data_path))
    print(file_names)
    for fidx, file_name in enumerate(file_names):
        if fidx in indices or (len(indices) == 0 and fidx + 1 == len(file_names)):
            data_for_file = pickle.load(open(os.path.join(data_path, file_name), 'rb'))
            yield data_for_file


def load_compressed_data(data_path, indices=[], all=False):
    file_names = sorted(os.listdir(data_path))
    print(file_names)
    for fidx, file_name in enumerate(file_names):
        if fidx in indices or (len(indices) == 0 and fidx + 1 == len(file_names)) or all:
            data_for_file = compressed_read(os.path.join(data_path, file_name))
            yield data_for_file


def load_behavioral_data(data_path, update_num=None, all=False):
    indices = [] if update_num is None else [update_num]
    state_data = load_compressed_data(
        data_path,
        indices=indices,
        all=all,
    )
    return state_data


def parse_loaded_session(d, env_idx):

    features = [
        'current_patch_num',
        'reward_site_idx',
        'current_reward_site_attempted',
        'agent_in_patch',
        'patch_reward_param',
        'action',
        'reward',
        'obs'
    ]

    features_to_time_series_dict = {}
    for f in features:
        if f == 'obs':
            features_to_time_series_dict[f] = np.zeros((len(d), 4))
        else:
            features_to_time_series_dict[f] = np.zeros((len(d)))
    
    for k in np.arange(len(d)):
        for f in features:
            features_to_time_series_dict[f][k] = d[k][f][env_idx]

    features_to_time_series_dict['current_patch_num'] = features_to_time_series_dict['current_patch_num'].astype(int)

    dwell_time = np.zeros((len(d)))
    rewards_seen_in_patch = np.zeros((len(d)))
    num_patch_types = len(np.unique(features_to_time_series_dict['current_patch_num']))
    rewards_seen_in_patch_type = np.zeros((len(d), num_patch_types))
    total_stops_in_patch_type = np.zeros((len(d), num_patch_types))
    
    for idx in np.arange(0, len(d)):
        if idx > 0 and features_to_time_series_dict['action'][idx] == 0:
            dwell_time[idx] = dwell_time[idx-1] + 1
        else:
            dwell_time[idx] = 0

        if features_to_time_series_dict['agent_in_patch'][idx]:
            if idx > 0:
                rewards_seen_in_patch[idx] = rewards_seen_in_patch[idx-1] + features_to_time_series_dict['reward'][idx]
            else:
                rewards_seen_in_patch[idx] = features_to_time_series_dict['reward'][idx]

        if idx > 0:
            curr_patch_type = features_to_time_series_dict['current_patch_num'][idx]
            rewards_seen_in_patch_type[idx, :] = rewards_seen_in_patch_type[idx-1, :]
            rewards_seen_in_patch_type[idx, curr_patch_type] += features_to_time_series_dict['reward'][idx]
            total_stops_in_patch_type[idx, :] = total_stops_in_patch_type[idx-1, :]
            if features_to_time_series_dict['current_reward_site_attempted'][idx] and not features_to_time_series_dict['current_reward_site_attempted'][idx-1]:
                total_stops_in_patch_type[idx, curr_patch_type] += 1

    features_to_time_series_dict['dwell_time'] = dwell_time
    features_to_time_series_dict['rewards_seen_in_patch'] = rewards_seen_in_patch
    features_to_time_series_dict['rewards_seen_in_patch_type'] = rewards_seen_in_patch_type
    features_to_time_series_dict['total_stops_in_patch_type'] = total_stops_in_patch_type
    
    return features_to_time_series_dict


def parse_session(data_path, env_idx):
    state_data = load_behavioral_data(
        data_path,
    )
    d = state_data.__next__()
    return parse_loaded_session(d, env_idx)


def parse_all_sessions(data_path, num_envs):
    state_data = load_behavioral_data(
        data_path,
        all=True,
    )

    all_time_series_dicts = []
    for d in state_data:
        for env_idx in range(num_envs):
            all_time_series_dicts.append(parse_loaded_session(d, env_idx))

    return all_time_series_dicts


# Load hidden states and behavior of network from `load path`
def load_hidden_and_behavior(load_path):
    data = load_numpy(os.path.join(load_path, 'hidden_state/*.npy').replace('\\','/'))
    data = np.transpose(data, [2, 1, 0])
    
    flattened_data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], order='C')
    
    pca = PCA()
    pc_activities = pca.fit_transform(flattened_data.T)
    pc_activities = pc_activities.T.reshape(data.shape, order='C')
    
    all_session_data = parse_all_sessions(
        os.path.join(load_path, 'state'),
        30,
    )

    return data, pc_activities, all_session_data, flattened_data, pca


def gen_alignment_chart(w, vs, vlim=None, title='', bias=None, ylabel='PC', scale=0.6):
    if bias is None:
        bias = np.zeros_like(w)
    if vlim is None:
        vlim = vs.shape[0]
    vs = vs[:vlim, :]
    w_norm = np.linalg.norm(w)
    
    # Set the figure and axes
    width = 3 * scale
    if (np.sum(bias) == 0):
        fig, axs = plt.subplots(1, 1, figsize=(width, 4 * scale), sharex=True, sharey=True)
        axs = [axs]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(2 * width, 4 * scale), sharex=True, sharey=True)

    # Define indices for the bars
    indices = np.arange(np.minimum(vlim, len(vs)))
    
    # Calculate the alignments
    print(vs.shape)
    print(w.shape)
    alignments = np.dot(w / w_norm, vs.T)
    print(alignments)
    
    # Create the horizontal bar chart
    axs[0].barh(indices, alignments, height=0.7)
    # Set the title with a specific font weight
    axs[0].set_title(f'{title}\nLength: {w_norm:.2f}', fontsize=12, fontweight='bold', pad=20)
    
    if not (np.sum(bias) == 0):
        axs[1].barh(indices, np.dot(vs, (w - bias) / np.linalg.norm(w - bias)), height=0.7)
        axs[1].set_title(f'After bias', fontsize=12, fontweight='bold', pad=20)

    # Customize the axes labels and limits
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-0.5, vlim - 0.5)

    for k in range(len(axs)):
        # Hide the top and right spines for a cleaner look
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)
    
        # Thicken the axes lines (left, bottom)
        axs[k].spines['left'].set_linewidth(2)
        axs[k].spines['bottom'].set_linewidth(2)
        
        # Add gridlines for better readability
        axs[k].grid(True, axis='x', linestyle='--', alpha=0.6)
    
        # Adjust ticks and labels
        axs[k].tick_params(axis='both', labelsize=12, length=6)
    
    axs[0].set_xlabel('Alignment')
    if ylabel is not None:
        axs[0].set_ylabel(ylabel)

    # Show the plot
    plt.tight_layout()

    return fig, axs


def find_patch_trajectories(agent_in_patch_ts):
    starts_and_stops = []
    start_idx = None
    for k in np.arange(len(agent_in_patch_ts)):
        if agent_in_patch_ts[k] > 0:
            if start_idx is None:
                start_idx = k
        elif (k > 0 and agent_in_patch_ts[k-1] > 0 and start_idx is not None):
            starts_and_stops.append(slice(start_idx, k))
            start_idx = None
    return starts_and_stops


def find_odor_site_trajectories(odor_site_indices, site_num=None):
    starts_and_stops = []
    start_idx = None
    for k in np.arange(len(odor_site_indices)):
        if (site_num is None and odor_site_indices[k] >= 0) or \
            odor_site_indices[k] == site_num:
            if start_idx is None:
                start_idx = k
        elif k > 0 and \
            ((odor_site_indices[k-1] >= 0 and site_num is None) or odor_site_indices[k-1] == site_num) and \
            start_idx is not None:
            starts_and_stops.append(slice(start_idx, k))
            start_idx = None
    return starts_and_stops


def find_trajectories_by_patch_type(session_data):
    patch_starts_and_ends = find_patch_trajectories(session_data['agent_in_patch'])
    trajs_by_patch_type = [[], [], []]
    for traj_idx, patch_traj_indices in enumerate(patch_starts_and_ends):
        patch_num = int(session_data['current_patch_num'][patch_traj_indices][0])
        trajs_by_patch_type[patch_num].append(patch_traj_indices)
    return trajs_by_patch_type


def find_odor_site_trajectories_by_patch_type(session_data, first_site_only=False, site_num=None):
    if first_site_only:
        site_num = 0
    patch_starts_and_ends = find_odor_site_trajectories(session_data['reward_site_idx'], site_num=site_num)
    trajs_by_patch_type = [[], [], []]
    for traj_idx, patch_traj_indices in enumerate(patch_starts_and_ends):
        patch_num = int(session_data['current_patch_num'][patch_traj_indices][0])
        trajs_by_patch_type[patch_num].append(patch_traj_indices)
    return trajs_by_patch_type