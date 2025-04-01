from .patch import Patch

class PatchType:

    def __init__(self, reward_site_len, interreward_site_len_mean, reward_func, odor_num, reward_func_param=None):
        self.reward_site_len = reward_site_len
        self.interreward_site_len_mean = interreward_site_len_mean
        self.reward_func = reward_func
        self.odor_num = odor_num
        self.reward_func_param = reward_func_param


    def generate_patch(self, patch_start):
        return Patch(
            patch_start,
            self.reward_site_len,
            self.interreward_site_len_mean,
            self.reward_func,
            self.odor_num,
            self.reward_func_param,
        )


    

    

    

