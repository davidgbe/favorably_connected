if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path
    curr_file_path = Path(__file__)
    print(curr_file_path)
    sys.path.append(str(curr_file_path.parent.parent))

import numpy as np
import glob2 as glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch.distributions.utils import logits_to_probs
from sklearn.decomposition import PCA
from agents.networks.a2c_rnn_split_vanilla import A2CRNN
from copy import deepcopy as copy
from nb_analysis_tools import load_numpy, load_compressed_data, parse_all_sessions, gen_alignment_chart, find_odor_site_trajectories_by_patch_type
from aux_funcs import compressed_write, logical_and, format_plot, compressed_read, sample_truncated_exp
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score
from numpy.linalg import norm
from imblearn.under_sampling import RandomUnderSampler
from torch.distributions.categorical import Categorical
from environments.treadmill_session import TreadmillSession
from environments.components.patch_type import PatchType
from agents.a2c_recurrent_agent_split import A2CRecurrentAgent


DATA_BASE_DIR = 'results/rl_agent_outputs'
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 1
ACTION_SIZE = 2
HIDDEN_SIZE = 128
NUM_ENVS = 1
DEVICE = 'cuda'

plt.rcParams['font.family'] = 'Helvetica Light'

# Load weights from given `load_path`
def load_network(load_path):
    
    network = A2CRNN(
        input_size=OBS_SIZE + ACTION_SIZE + 1,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        device='cuda',
        var_noise=0,
    )
    
    saved_checkpoint = torch.load(os.path.join(DATA_BASE_DIR, load_path).replace('\\','/'), weights_only=True)
    if 'network_state_dict' in saved_checkpoint:
        network.load_state_dict(saved_checkpoint['network_state_dict'])
    else:
        network.load_state_dict(saved_checkpoint)
    network.eval()

    return network


# Load hidden states and behavior of network from `load path`
def load_hidden_and_behavior(load_path):
    data = load_numpy(os.path.join(DATA_BASE_DIR, load_path, 'hidden_state/*.npy').replace('\\','/'))
    data = np.transpose(data, [2, 1, 0])
    
    flattened_data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], order='C')
    
    pca = PCA()
    pc_activities = pca.fit_transform(flattened_data.T)
    pc_activities = pc_activities.T.reshape(data.shape, order='C')
    
    all_session_data = parse_all_sessions(
        os.path.join(DATA_BASE_DIR, load_path, 'state'),
        30,
    )

    return data, pc_activities, all_session_data, pca


weight_paths = [
    'he_init_vanilla_inoise_0p05_2025-07-06_15_02_35_603284_var_noise_0.0001_activity_weight_1/rnn_weights/00950.pth',
]

hidden_and_behavior_paths = [
    'test_he_init_linear_437044_2400_2025-07-07_14_22_12_308938_var_noise_0.0_activity_weight_1',
]

for i, (weight_path, hidden_and_behavior_path) in enumerate(zip(weight_paths, hidden_and_behavior_paths)):
    network = load_network(weight_path)
    # acc_reward_vec_path = os.path.join(DATA_BASE_DIR, Path(weight_path).parents[1], 'stored_pcs_and_weights/rewards_seen_in_patch.pkl').replace('/', '\\')
    # acc_reward_vec = compressed_read(acc_reward_vec_path)
    hidden_activities, pc_activities, all_session_data, pca = load_hidden_and_behavior(hidden_and_behavior_path)

def extract_state_on_visual_cue(hidden_activity, session_data):
    visual_cue_went_high_mask = logical_and(
        session_data['obs'][:, 0] > 0,
        np.roll(session_data['obs'][:, 0], 1) < 0.5,
    )
    return hidden_activity[:, visual_cue_went_high_mask]

visual_cue_states = extract_state_on_visual_cue(hidden_activities[:, 0, :], all_session_data[0])

flattened_ha = hidden_activities.reshape(hidden_activities.shape[0], hidden_activities.shape[1] * hidden_activities.shape[2], order='C')

print(flattened_ha.shape)

pca = PCA()
pc_activities = pca.fit_transform(flattened_ha.T)
pc_activities = pc_activities.T.reshape(hidden_activities.shape, order='C')

window = slice(0, 400)
trial_index = 3

scale = 1
fig, axs = plt.subplots(2, 1, figsize=(6 * scale, 6 * scale), sharex=True)
axs[1].matshow(pc_activities[:6, trial_index, window], cmap='bwr', vmin=-0.8, vmax=0.8, aspect=10)
axs[1].set_ylabel('PC')
axs[1].set_xlabel('Time')

inputs = np.concatenate([all_session_data[trial_index]['obs'][window, :], all_session_data[trial_index]['action'][window, np.newaxis], all_session_data[trial_index]['reward'][window, np.newaxis]], axis=1).T
axs[0].matshow(inputs, cmap='bwr', vmin=-1, vmax=1, aspect=10)
axs[0].set_yticks(np.arange(6), ['Visual cue', 'Odor (unr.)', 'Odor (low rew.)', 'Odor (high rew.)', 'Action', 'Reward'])
axs[0].set_xlabel('Time')

format_plot(axs)
fig.tight_layout()
fig.savefig('results/figures/_one.png')
plt.close()

def run_input_exp(network, t_steps=300, action_feedback=False,):
    np.random.seed(34)
    # ENVIRONEMENT PARAMS
    PATCH_TYPES_PER_ENV = 3
    OBS_SIZE = PATCH_TYPES_PER_ENV + 1
    ACTION_SIZE = 2
    DWELL_TIME_FOR_REWARD = 6
    # for actual task, reward sites are 50 cm long and interreward sites are between 20 and 100, decay rate 0.05 (truncated exp.)
    REWARD_SITE_LEN = 3
    INTERREWARD_SITE_LEN_BOUNDS = [1, 6]
    INTEREWARD_SITE_LEN_DECAY_RATE = 0.8
    REWARD_DECAY_CONSTS = [0, 10, 30]
    REWARD_PROB_PREFACTOR = 0.8
    # for actual task, interpatch lengths are 200 to 600 cm, decay rate 0.01 
    INTERPATCH_LEN_BOUNDS = [1, 12]
    INTERPATCH_LEN_DECAY_RATE = 0.1
    INPUT_NOISE_STD = 0

    if network.hidden_states is not None:
        network.reset_state()

    def interreward_site_transition():
        return int(sample_truncated_exp(
            size=1,
            bounds=INTERREWARD_SITE_LEN_BOUNDS,
            decay_rate=INTEREWARD_SITE_LEN_DECAY_RATE,
        )[0])
    
    
    def interpatch_transition():
        return int(sample_truncated_exp(
            size=1,
            bounds=INTERPATCH_LEN_BOUNDS,
            decay_rate=INTERPATCH_LEN_DECAY_RATE,
        )[0])

    decay_consts_for_reward_funcs = [0, 10, 30]
    print('Begin stoch. treadmill')
    print(decay_consts_for_reward_funcs)

    patch_types = []
    for i in range(PATCH_TYPES_PER_ENV):
        decay_const_for_i = decay_consts_for_reward_funcs[i]
        active = (decay_const_for_i != 0)
        def reward_func(site_idx, decay_const_for_i=decay_const_for_i, active=active):
            c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i) if decay_const_for_i > 0 else 0
            if np.random.rand() < c and active:
                return 1
            else:
                return 0
        patch_types.append(
            PatchType(
                reward_site_len=REWARD_SITE_LEN,
                interreward_site_len_func=interreward_site_transition,
                reward_func=reward_func,
                odor_num=i,
                reward_func_param=(decay_consts_for_reward_funcs[i] if active else 0.0),
            )
        )

    transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

    env = TreadmillSession(
        patch_types=patch_types,
        transition_mat=transition_mat,
        interpatch_len_func=interpatch_transition,
        dwell_time_for_reward=DWELL_TIME_FOR_REWARD,
        obs_size=PATCH_TYPES_PER_ENV + 1,
        verbosity=False,
    )

    agent = A2CRecurrentAgent(
        network,
        action_space_dims=ACTION_SIZE,
        n_envs=NUM_ENVS,
        device=DEVICE,
        critic_weight=0, # changed for Optuna
        entropy_weight=0, # changed for Optuna
        gamma=0, # changed for Optuna
        learning_rate=0, # changed for Optuna
        activity_weight=1,
        input_noise_std=INPUT_NOISE_STD,
    )

    all_hidden_out = torch.empty((t_steps, visual_cue_states.shape[0], 1))
    action_probs = torch.empty((t_steps, ACTION_SIZE, 1))
    action_probs_totals = torch.zeros((t_steps, ACTION_SIZE, 1))

    obs, info = env.reset()

    inputs = np.zeros((t_steps, 1, 7))
    # with torch.no_grad():
    for i in range(t_steps):
        obs_tensor = torch.tensor(obs[None, :]).float()
        action = agent.sample_action(obs_tensor).clone().detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action[0])
        agent.append_reward(torch.tensor(reward).unsqueeze(0))

        inputs[i, :, :4] = obs
        inputs[i, :, 5] = action[0]

        all_hidden_out[i, :, 0] = agent.activities[-1]
        # if i == 0:
        #     input_i = torch.concatenate([torch.tensor(obs), torch.zeros((3,))]).float()
        # else:
        #     actions = torch.zeros((2,))
        #     actions[action] = 1
        #     input_i = torch.concatenate([torch.tensor(obs), actions, torch.zeros((1,))]).float()
        # inputs[i, ...] = input_i
        # input_i_cuda = input_i[None, :].to(DEVICE)
        # action_logits, value, hidden_out = network(input_i_cuda)
        # all_hidden_out[i, ...] = hidden_out.clone().cpu().T

        # distrib = Categorical(logits=action_logits)
        # action = distrib.sample().cpu()
        # action_probs[i, ...] = distrib.probs.Tsss
        # action_probs_totals[i, ...] += action_probs[i, ...]

        # obs, reward, terminated, truncated, info = env.step(action)

    return {
        'all_hidden_out': all_hidden_out.detach(),
        'action_probs': action_probs.detach(),
        'action_probs_totals': action_probs_totals.detach(),
        'inputs': inputs,
    }


def plot_exp(network, pca, t_steps=300, action_feedback=False, selected_trial=0, window=None):
    d = run_input_exp(
        network,
        # inputs,
        t_steps=t_steps,
        action_feedback=action_feedback,
    )
    
    scale = 0.8
    fig, axs = plt.subplots(6, 1, figsize=(9 * scale, 11 * scale), sharex=True)
    axs[1].matshow(d['all_hidden_out'][..., selected_trial].T, aspect='auto', cmap='bwr', vmin=-0.25, vmax=0.25)

    axs[2].matshow(pca.transform(d['all_hidden_out'][..., selected_trial]).T[:6, :], aspect='auto', cmap='bwr', vmin=-0.8, vmax=0.8)

    axs[0].matshow(d['inputs'][:, 0, np.array([0, 1, 2, 3, 4, 5, 6])].T, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
        
    axs[3].matshow(d['action_probs'][..., selected_trial].T, aspect='auto', cmap='bwr', vmin=-1, vmax=1)

    axs[5].plot(np.arange(t_steps), d['action_probs_totals'][:, 0, selected_trial])

    if window:
        axs[0].set_xlim(*window)

    axs[0].set_yticks(np.arange(7), ['Visual cue', 'Odor (unr.)', 'Odor (low rew.)', 'Odor (high rew.)', 'Stay', 'Run', 'Reward'])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Input')
    
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Unit activity\n(RNN)')

    fig.savefig('results/figures/_two.png')
    plt.close()

t_steps = 400
plot_exp(network, pca, action_feedback=True, t_steps=t_steps)