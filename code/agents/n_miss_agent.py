import numpy as np

class NMissAgent:
    """
    Heuristic agent that waits for a reward after detecting odor cues.
    It leaves a patch only after N consecutive missed rewards.
    """

    def __init__(self, n_envs, wait_time_for_reward, odor_cues_indices, n_misses=3):
        self.wait_time_for_reward = wait_time_for_reward
        self.rewards = []
        self.odor_cues_start = odor_cues_indices[0]
        self.odor_cues_end = odor_cues_indices[1]
        self.last_observations = None
        self.dwell_time = np.zeros((n_envs,), dtype=int)
        self.missed_rewards = np.zeros((n_envs,), dtype=int)  # track consecutive misses
        self.n_envs = n_envs
        self.n_misses = n_misses

    def sample_action(self, observations, tau=None):
        if tau is not None:
            self.n_misses = tau
            # print(self.n_misses)
        past_reward = self.get_last_reward()
        # print('obs', observations)
        # print('dwell_time', self.dwell_time)
        # print('reward', past_reward)
        # print('consecutive_missed_rewards', self.missed_rewards)

        # Reset dwell time and missed counter if reward received
        reward_received = past_reward > 0
        odor_on = (observations[:, self.odor_cues_start:self.odor_cues_end + 1] > 0.5).any(axis=1)
        in_patch = (observations[:, 0] > 0.5)
        n_misses_in_patch = self.n_misses <= self.missed_rewards 

        if self.last_observations is not None:
            obs_comp_vec = (
                observations[:, self.odor_cues_start:self.odor_cues_end + 1]
                > self.last_observations[:, self.odor_cues_start:self.odor_cues_end + 1]
            )
            odor_detected = (obs_comp_vec > 0.5).any(axis=1)

            self.dwell_time = np.where(
                odor_detected, self.wait_time_for_reward, self.dwell_time
            )

        else:
            odor_detected = np.zeros(observations.shape[0]).astype(bool)

        self.dwell_time = np.where(in_patch & odor_on, self.dwell_time - 1, self.dwell_time)

        action = np.where(
            in_patch & odor_on & (self.dwell_time > 2) & (not n_misses_in_patch),
            0,
            1,
        )

        self.missed_rewards = np.where(
            in_patch,
            np.where(
                odor_on,
                np.where(
                    self.dwell_time == 0,
                    np.where(
                        reward_received,
                        0, # if reward received, reset count
                        self.missed_rewards + 1, # increment count if we miss a reward
                    ),
                    self.missed_rewards, # keep dwell time the same if dwell time isn't zero
                ),        
                self.missed_rewards, # leave unchanged if animal not in odor site
            ),
            0, # set to zero if agent is out of patch
        )

        # print('action', action)
        # print()

        self.last_observations = observations.copy()
        return action

    def get_last_reward(self):
        return self.rewards[-1] if len(self.rewards) > 0 else np.zeros((self.n_envs,))

    def append_reward(self, reward):
        self.rewards.append(reward)

    def reset_state(self):
        self.rewards = []
        self.last_observations = None
        self.dwell_time = np.zeros((self.n_envs,), dtype=int)
        self.missed_rewards = np.zeros((self.n_envs,), dtype=int)