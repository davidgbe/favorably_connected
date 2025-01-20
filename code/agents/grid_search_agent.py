import numpy as np
from copy import deepcopy as copy

class GridSearchAgent():

    def __init__(self, n_envs, wait_time_for_reward, odor_cues_indices, patch_cue_idx, stop_ranges_per_patch):
        self.wait_time_for_reward = wait_time_for_reward
        self.rewards = []
        self.odor_cues_start = odor_cues_indices[0]
        self.odor_cues_end = odor_cues_indices[1]
        self.patch_cue_idx = patch_cue_idx
        self.stop_ranges_per_patch = stop_ranges_per_patch # (n_patches, 2)

        self.last_observations = None
        self.dwell_time = np.zeros((n_envs), dtype=int)
        self.odor_site_idx = np.zeros((n_envs), dtype=int)
        self.current_patch_type = np.zeros((n_envs), dtype=int)
        self.n_envs = n_envs

        n_patches = self.odor_cues_end - self.odor_cues_start
        self.reward_rates_for_policies = np.zeros((n_envs, (stop_ranges_per_patch.max() + 1) ** n_patches))
        self.n_stops_for_policies = np.zeros((n_patches, (stop_ranges_per_patch.max() + 1) ** n_patches), dtype=int)
        self.n_stops_for_patch = copy(stop_ranges_per_patch[:, 0])
        self.optimized_n_stops_for_patch = np.zeros((n_envs, n_patches), dtype=int)

        self.search_finished = False
        self.policy_idx = 0

    def sample_action(self, observations):
        if self.last_observations is None:
            self.last_observations = np.zeros(observations.shape)

        last_patch_start_cues = self.last_observations[:, self.patch_cue_idx]
        last_odor_cues = self.last_observations[:, self.odor_cues_start:self.odor_cues_end]

        patch_start_cues = observations[:, self.patch_cue_idx]
        odor_cues = observations[:, self.odor_cues_start:self.odor_cues_end]

        patch_start_cues_comp = patch_start_cues - last_patch_start_cues
        odor_cues_comp = odor_cues - last_odor_cues

        self.odor_site_idx = np.where(patch_start_cues_comp > 0, 0, self.odor_site_idx)
        self.current_patch_type = np.where(odor_cues.sum(axis=1) > 0, odor_cues.argmax(axis=1), self.current_patch_type)
        current_odor_cue_off = np.array([odor_cues_comp[k, self.current_patch_type[k]] for k in range(self.n_envs)]) < 0
        self.odor_site_idx = np.where(current_odor_cue_off, self.odor_site_idx + 1, self.odor_site_idx)

        if self.search_finished:
            stops_for_patch = np.empty((self.n_envs))
            for k in np.arange(self.n_envs):
                stops_for_patch[k] = self.optimized_n_stops_for_patch[k, self.current_patch_type[k]]
        else:
            stops_for_patch = self.n_stops_for_patch[self.current_patch_type]

        odor_site_entered = np.array([odor_cues_comp[k, self.current_patch_type[k]] for k in range(self.n_envs)]) > 0

        should_stop = np.logical_and(self.odor_site_idx < stops_for_patch, odor_site_entered)
        self.dwell_time = np.where(should_stop, self.wait_time_for_reward, self.dwell_time)

        action = np.where(self.dwell_time > 0, 0, 1)
        self.dwell_time = np.where(self.dwell_time > 0, self.dwell_time - 1, 0)

        self.last_observations = observations.copy()
        return action


    def get_last_reward(self):
        return self.rewards[-1] if len(self.rewards) > 0 else np.zeros((self.n_envs))


    def append_reward(self, reward):
        self.rewards.append(reward)

    
    def cue_session_end(self):
        if not self.search_finished:
            rewards = np.array(self.rewards)
            avg_reward_rates = np.mean(self.rewards, axis=0)

            self.n_stops_for_policies[:, self.policy_idx] = self.n_stops_for_patch.copy()
            self.reward_rates_for_policies[:, self.policy_idx] = avg_reward_rates

            print('Policy end:')
            print('Index', self.policy_idx)
            print(self.n_stops_for_policies[:, self.policy_idx])
            print('Reward rates', self.reward_rates_for_policies[:, self.policy_idx])

            idx = 0
            if (self.n_stops_for_patch == self.stop_ranges_per_patch[:, 1]).all():
                self.search_finished = True
                for k in range(self.n_envs):
                    print(self.reward_rates_for_policies[k, :])
                    optimal_policy_idx = self.reward_rates_for_policies[k, :].argmax()
                    self.optimized_n_stops_for_patch[k, :] = self.n_stops_for_policies[:, optimal_policy_idx]
                    print(self.optimized_n_stops_for_patch)
            else:
                idx = 0
                while idx < len(self.n_stops_for_patch):
                    if self.n_stops_for_patch[idx] < self.stop_ranges_per_patch[idx, 1]:
                        self.n_stops_for_patch[idx] += 1
                        break
                    else:
                        self.n_stops_for_patch[idx] = self.stop_ranges_per_patch[idx, 0]
                        idx += 1

            self.policy_idx += 1
            

    def reset_state(self):
        self.rewards = []
        self.last_observations = None
        self.dwell_time = np.zeros((self.n_envs))