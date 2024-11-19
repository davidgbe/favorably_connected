import numpy as np

class Patch:

    def __init__(self, n_reward_sites, reward_site_len, interreward_site_len_mean, reward_func, odor_num, reward_func_param=None):
        self.n_reward_sites = n_reward_sites
        self.reward_site_len = reward_site_len
        self.interreward_site_len_mean = interreward_site_len_mean
        self.reward_func = reward_func
        self.odor_num = odor_num
        self.reward_func_param = reward_func_param


    def get_bounds(self, patch_start):
        interreward_site_lens = 1 + np.random.poisson(lam=self.interreward_site_len_mean - 1, size=self.n_reward_sites)
        patch_len = np.sum(interreward_site_lens) + self.reward_site_len * self.n_reward_sites
        reward_bounds = []
        patch_bounds = np.array([0, patch_len]) + patch_start
        for i in range(self.n_reward_sites):
            reward_site_bounds = np.array([i * self.reward_site_len + np.sum(interreward_site_lens[:(i + 1)]), (i + 1) * self.reward_site_len + np.sum(interreward_site_lens[:(i + 1)])]) + patch_start
            reward_bounds.append(reward_site_bounds)
        return patch_bounds, reward_bounds


    def get_reward(self, reward_site_idx):
        return self.reward_func(reward_site_idx)

    
    def get_odor_num(self):
        return self.odor_num

    