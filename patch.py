import numpy as np

class Patch:

    def __init__(self, n_reward_sites, reward_site_len, interreward_site_len, reward_func, patch_num):
        self.n_reward_sites = n_reward_sites
        self.reward_site_len = reward_site_len
        self.interreward_site_len = interreward_site_len
        self.r_and_ir_len = reward_site_len + interreward_site_len
        self.patch_len = (interreward_site_len + reward_site_len) * n_reward_sites
        self.reward_func = reward_func
        self.patch_num = patch_num


    def get_patch_bounds(self, patch_start):
        return np.array([0, self.patch_len]) + patch_start


    def get_reward_bounds(self, patch_start):
        reward_bounds = []
        for i in range(self.n_reward_sites):
            reward_site_bounds = np.array([i * self.r_and_ir_len, i * self.r_and_ir_len + self.reward_site_len]) + patch_start + self.interreward_site_len
            reward_bounds.append(reward_site_bounds)
        return reward_bounds


    def get_reward(self, reward_site_idx):
        return self.reward_func(reward_site_idx)

    