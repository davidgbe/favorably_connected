import numpy as np

class Patch:

    def __init__(self, patch_start, n_reward_sites, reward_site_len, interreward_site_len_mean, reward_func, odor_num, reward_func_param=None):
        self.patch_start = patch_start
        self.n_reward_sites = n_reward_sites
        self.reward_site_len = reward_site_len
        self.interreward_site_len_mean = interreward_site_len_mean
        self.reward_func = reward_func
        self.odor_num = odor_num
        self.reward_func_param = reward_func_param
        self.current_reward_site_idx = None
        self.generate_reward_site()
        self.all_reward_sites = []


    def generate_reward_site(self)
        interreward_site_len = 1 + np.random.poisson(lam=self.interreward_site_len_mean - 1)
        if self.current_reward_site_start is None:
            self.current_reward_site_start = self.patch_start + interreward_site_len
            self.current_reward_site_idx = 0
        else:
            self.current_reward_site_start = self.current_reward_site_start + self.reward_site_len + interreward_site_len
            self.current_reward_site_idx += 1
        self.all_reward_site_starts.append([
            self.current_reward_site_start,
            self.current_reward_site_start + self.reward_site_len,
            self.current_reward_site_start + self.reward_site_len + 
        ])


    def contains_position(self, pos):
        return (pos >= self.patch_start) and (pos < (self.all_reward_site_starts[-1] + self.reward_site_len))
        

    def get_odor_site_idx_of_pos(self, pos):
        idx = len(self.all_reward_site_starts) - 1
        for rw_site_start in reversed(self.all_reward_site_starts):
            if idx < 0:
                return -1
            if pos > rw_site_start + self.reward_site_len:
                return -1
            elif pos >= rw_site_start and pos < rw_site_start + self.reward_site_len:
                return idx
            else:
                idx -= 1


    def get_reward(self, pos):
        if pos >= self.current_reward_site_start and pos < self.current_reward_site_start + self.reward_site_len:
            return self.reward_func(self.current_reward_site_idx)
        else:
            return 0


    def get_odor_num(self):
        return self.odor_num



class PatchType:

    def __init__(self, n_reward_sites, reward_site_len, interreward_site_len_mean, reward_func, odor_num, reward_func_param=None):
        self.n_reward_sites = n_reward_sites
        self.reward_site_len = reward_site_len
        self.interreward_site_len_mean = interreward_site_len_mean
        self.reward_func = reward_func
        self.odor_num = odor_num
        self.reward_func_param = reward_func_param


    def generate_patch_instance(self, patch_start):
        return Patch(
            patch_start,
            self.n_reward_sites,
            self.reward_site_len,
            self.interreward_site_len_mean,
            self.reward_func,
            self.odor_num,
            self.reward_func_param,
        )
    
    
    def get_odor_num(self):
        return self.odor_num

    