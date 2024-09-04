import numpy as np

class WSLSAgent():

    def __init__(self, n_envs, wait_time_for_reward, odor_cues_indices, patch_cue_idx):
        self.wait_time_for_reward = wait_time_for_reward
        self.rewards = []
        self.odor_cues_start = odor_cues_indices[0]
        self.odor_cues_end = odor_cues_indices[1]
        self.patch_cue_idx = patch_cue_idx
        self.stay_in_current_patch = np.zeros((n_envs), dtype=bool)
        self.last_observations = None
        self.dwell_time = np.zeros((n_envs), dtype=int)
        self.n_envs = n_envs


    def sample_action(self, observations):
        self.stay_in_current_patch = np.where(observations[:, self.patch_cue_idx] > 0, True, self.stay_in_current_patch)
        past_reward = self.get_last_reward()

        self.stay_in_current_patch = np.where(self.dwell_time == 1, False, self.stay_in_current_patch)
        self.dwell_time = np.where(past_reward > 0, 0, self.dwell_time)
        action = np.where(np.logical_and(self.dwell_time > 0, self.stay_in_current_patch), 0, 1)
        self.dwell_time = np.where(self.dwell_time > 0, self.dwell_time - 1, 0)

        if self.last_observations is not None:
            obs_comp_vec = observations[:, self.odor_cues_start:self.odor_cues_end] > self.last_observations[:, self.odor_cues_start:self.odor_cues_end]
            obs_changed_while_staying = np.logical_and((obs_comp_vec > 0).any(axis=1), self.stay_in_current_patch)
            self.dwell_time = np.where(obs_changed_while_staying, self.wait_time_for_reward, self.dwell_time)
            action = np.where(obs_changed_while_staying, 0, action)

        self.last_observations = observations.copy()
        return action


    def get_last_reward(self):
        return self.rewards[-1] if len(self.rewards) > 0 else np.zeros((self.n_envs))


    def append_reward(self, reward):
        self.rewards.append(reward)


    def reset_state(self):
        self.rewards = []
        self.last_observations = None
        self.dwell_time = np.zeros((self.n_envs))