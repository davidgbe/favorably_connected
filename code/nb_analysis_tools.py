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
from copy import deepcopy as copy
import pandas as pd


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
            print(file_name)
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


def parse_behavioral_data(d, env_idx):

    features = [
        'current_patch_num',
        'current_patch_start',
        'current_position',
        'reward_bounds',
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
        features_to_time_series_dict[f] = []
        
    for k in np.arange(len(d)):
        for f in features:
            features_to_time_series_dict[f].append(d[k][f][env_idx])
            
    for f in features:
        try:
            features_to_time_series_dict[f] = np.array(features_to_time_series_dict[f])
        except e:
            print(e)

    features_to_time_series_dict['current_patch_num'] = features_to_time_series_dict['current_patch_num'].astype(int)
    
    num_patch_types = len(np.unique(features_to_time_series_dict['current_patch_num']))

    dwell_times = np.zeros((len(d)))
    rewards_seen_in_patch = np.zeros((len(d)))
    rewards_seen_in_patch_type = np.zeros((len(d), num_patch_types))
    total_stops_in_patch_type = np.zeros((len(d), num_patch_types))
    dwell_time = np.zeros((len(d)))

    dwell_times_at_positions = []
    rewards_at_positions = [0]
    reward_attempted_at_positions = [False]
    dwell_time = 1
    last_p = None
    unique_patch_params = np.zeros((3))
    
    for idx, p in enumerate(features_to_time_series_dict['current_position']):

        if last_p is not None:
            if (p != last_p):
                rewards_at_positions.append(0)
                reward_attempted_at_positions.append(False)
                dwell_times_at_positions.append([dwell_time])
                dwell_time = 1
            else:
                dwell_time += 1

        dwell_times[idx] = dwell_time
        rewards_at_positions[-1] += features_to_time_series_dict['reward'][idx]
        reward_attempted_at_positions[-1] = True if features_to_time_series_dict['current_reward_site_attempted'][idx] else reward_attempted_at_positions[-1]
        last_p = p

        if features_to_time_series_dict['agent_in_patch'][idx]:
            curr_patch_type = features_to_time_series_dict['current_patch_num'][idx]
            unique_patch_params[curr_patch_type] = features_to_time_series_dict['patch_reward_param'][idx]
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

    features_to_time_series_dict['dwell_time'] = dwell_times
    features_to_time_series_dict['rewards_seen_in_patch'] = rewards_seen_in_patch
    features_to_time_series_dict['rewards_seen_in_patch_type'] = rewards_seen_in_patch_type
    features_to_time_series_dict['total_stops_in_patch_type'] = total_stops_in_patch_type
    features_to_time_series_dict['dwell_times_at_positions'] = np.array(dwell_times_at_positions)
    features_to_time_series_dict['rewards_at_positions'] = np.array(rewards_at_positions)
    features_to_time_series_dict['reward_attempted_at_positions'] = np.array(reward_attempted_at_positions)
    features_to_time_series_dict['unique_patch_reward_params'] = unique_patch_params

    
    return features_to_time_series_dict


def parse_session(data_path, env_idx):
    state_data = load_behavioral_data(
        data_path,
    )
    d = state_data.__next__()
    return parse_behavioral_data(d, env_idx)


def parse_all_sessions(data_path, num_envs):
    state_data = load_behavioral_data(
        data_path,
        all=True,
    )

    all_time_series_dicts = []
    for d in state_data:
        for env_idx in range(num_envs):
            all_time_series_dicts.append(parse_behavioral_data(d, env_idx))

    return all_time_series_dicts


def get_session_summaries(all_behavior_data, max_reward_sites=40, max_acc_reward=40):
    all_session_summaries = []

    for b_data in all_behavior_data:
        rewards_at_positions = b_data['rewards_at_positions']
        reward_attempted_at_positions = b_data['reward_attempted_at_positions']
        all_patch_nums = b_data['current_patch_num']
        all_patch_reward_params = b_data['patch_reward_param']
        rewards_seen_in_patch = b_data['rewards_seen_in_patch']

        ss = {
            'all_odor_site_data': [],
            'reward_param_of_stop': [],
            'site_idx_of_stop': [],
            'site_stops_for_patch_type': np.zeros((all_patch_nums.max() + 1, max_reward_sites)),
            'site_stop_opportunities_for_patch_type': np.zeros((all_patch_nums.max() + 1, max_reward_sites)),
            'acc_reward_stops_for_patch_type': np.zeros((all_patch_nums.max() + 1, max_acc_reward)),
            'acc_reward_stop_opportunities_for_patch_type': np.zeros((all_patch_nums.max() + 1, max_acc_reward)),
            'patches_entered_for_patch_type': np.zeros((all_patch_nums.max() + 1,)),
            'reward_param_for_patch_type': np.zeros((all_patch_nums.max() + 1,))
        }

        last_pstart = None
        last_reward_site_start = None
        last_reward_site_end = None
        last_odor_site_data = None
        patch_count = 0
        rw_site_counter = 0

        for i, pstart in enumerate(b_data['current_patch_start']):
            if last_pstart is None or (pstart != last_pstart).any():
                pt = all_patch_nums[i]
                ss['patches_entered_for_patch_type'][pt] += 1
                ss['reward_param_for_patch_type'][pt] = all_patch_reward_params[i]
                patch_count += 1
                rw_site_counter = 0
                if last_odor_site_data is not None and last_odor_site_data['added'][0] == 0:
                    ss['all_odor_site_data'].append(last_odor_site_data)
                    last_odor_site_data['added'] = [1]
                    last_odor_site_data = None

            rwsb = copy(b_data['reward_bounds'][i])
            reward_site_start = int(rwsb[0])

            if last_reward_site_start is None or not np.isclose(last_reward_site_start, reward_site_start):
                odor_site_data = pd.DataFrame({
                    'dist_last_odor_site': [np.nan],
                    'patch_reward_param': [all_patch_reward_params[i]],
                    'index': [rw_site_counter],
                    'stopped': [0],
                    'rewarded': [1],
                    'rewarded_last_odor_site': [0],
                    'added': [0],
                    'rewards_seen_in_patch': [int(rewards_seen_in_patch[i])],
                })

                if last_odor_site_data is not None:
                    odor_site_data['dist_last_odor_site'] = reward_site_start - last_reward_site_end
                    odor_site_data['rewarded_last_odor_site'] = last_odor_site_data['rewarded']

                if rewards_seen_in_patch[i] < max_acc_reward:
                    ss['acc_reward_stop_opportunities_for_patch_type'][pt, int(rewards_seen_in_patch[i])] += 1

                if rw_site_counter < max_reward_sites:
                    ss['site_stop_opportunities_for_patch_type'][pt, rw_site_counter] += 1

                if np.sum(reward_attempted_at_positions[int(rwsb[0]):int(rwsb[1])]) == 0:
                    pass  # pb handling omitted for brevity
                else:
                    reward = np.sum(rewards_at_positions[int(rwsb[0]):int(rwsb[1])])
                    ss['reward_param_of_stop'].append(all_patch_reward_params[i])
                    ss['site_idx_of_stop'].append(rw_site_counter)
                    odor_site_data['stopped'] = [1]
                    odor_site_data['rewarded'] = [reward]

                    if rw_site_counter < max_reward_sites:
                        ss['site_stops_for_patch_type'][pt, rw_site_counter] += 1

                    if rewards_seen_in_patch[i] < max_acc_reward:
                        ss['acc_reward_stops_for_patch_type'][pt, int(rewards_seen_in_patch[i])] += 1

                    rw_site_counter += 1

                if last_odor_site_data is not None:
                    ss['all_odor_site_data'].append(last_odor_site_data)
                    last_odor_site_data['added'] = [1]
                last_odor_site_data = odor_site_data

            last_pstart = pstart
            last_reward_site_start = reward_site_start
            last_reward_site_end = rwsb[1]

        all_session_summaries.append(ss)

    return all_session_summaries

    
def get_all_session_summaries(load_path, update_num=None):
    all_session_data = parse_all_sessions(
        os.path.join(load_path, 'state'),
        30,
    )
    print('Session data loaded')
    session_summaries = get_session_summaries(all_session_data)
    print('Session summaries generated')
    return session_summaries
        

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