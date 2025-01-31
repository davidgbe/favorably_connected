import numpy as np

class Patch:

    def __init__(
            self,
            patch_start,
            reward_site_len,
            interreward_site_len_mean,
            reward_func,
            odor_num,
            reward_func_param=None
        ):
        self.patch_start = patch_start
        self.reward_site_len = reward_site_len
        self.interreward_site_len_mean = interreward_site_len_mean
        self.reward_func = reward_func
        self.odor_num = odor_num
        self.reward_func_param = reward_func_param
        self.reward_sum = 0

        self.current_reward_site_idx = -1
        self.current_reward_site_bounds = None
        self.generate_reward_site()


    def generate_reward_site(self):
        interreward_site_len = 1 + np.random.poisson(lam=self.interreward_site_len_mean - 1)
        self.current_reward_site_idx += 1
        if self.current_reward_site_idx == 0:
            rws_start = self.patch_start + interreward_site_len
        else:
            rws_start = self.current_reward_site_bounds[1] + interreward_site_len
        self.current_reward_site_bounds = (rws_start, rws_start + self.reward_site_len)
        return self.current_reward_site_bounds, self.current_reward_site_idx

    
    def get_bounds(self):
        return self.patch_start, self.current_reward_site_bounds[1]


    def get_reward(self):
        r = self.reward_func(self.current_reward_site_idx)
        # r = self.reward_func(self.reward_sum)
        self.reward_sum += r
        return r

    
    def get_odor_num(self):
        return self.odor_num

    