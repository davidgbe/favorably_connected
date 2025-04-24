import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bisect import bisect

class TreadmillSession(gym.Env):

    def __init__(
        self,
        patch_types,
        transition_mat,
        interpatch_len,
        dwell_time_for_reward,
        obs_size,
        obs_noise_std=0,
        verbosity=True,
        first_patch_start=None,
        odor_lesioned=False,
    ):
        self.patch_types = patch_types
        self.current_patch = None
        self.current_patch_type = None
        self.current_patch_num = None
        self.transition_mat = transition_mat
        self.interpatch_len = interpatch_len
        self.dwell_time_for_reward = dwell_time_for_reward
        self.set_verbosity(verbosity)
        self.step_vals = np.array([0, 1])
        self.start_new_session(first_patch_start)
        self.odor_lesioned = odor_lesioned
        if type(obs_size) is not int:
            raise ValueError("'obs_size' must be an integer")
        if obs_size < len(patch_types):
            raise ValueError("'obs_size' must be at least equal to the number of patch types")
        self.obs_size = obs_size
        self.obs_noise_std = obs_noise_std
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.step_vals))


    # begin gymnasium specific API
    # note: these functions simply wrap other functions defined later on

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_new_session()

        return self.get_observations(), {}


    def step(self, action):
        forward_movement_amount = self.step_vals[action]
        info = self.get_info(forward_movement_amount)
        reward = self.move_forward(forward_movement_amount)
        obs = self.get_observations()

        info['reward'] = reward

        return obs, reward, False, False, info

    # end gymnasium API


    def get_info(self, action):
        return {
            'action': action,
            'obs': self.get_observations(),
            'current_patch_num': self.current_patch_num,
            'current_position': self.current_position,
            'current_patch_start': self.current_patch.patch_start,
            'reward_bounds': self.current_patch.current_reward_site_bounds,
            'agent_in_patch': self.is_agent_in_current_patch(),
            'reward_site_idx': self.get_reward_site_idx_of_current_pos(),
            'current_reward_site_attempted': self.current_reward_site_attempted,
            'patch_reward_param': self.current_patch.reward_func_param,
        }


    def start_new_session(self, first_patch_start=None):
        self.current_patch_num = np.random.randint(0, len(self.patch_types))
        self.current_position = 0
        self.wprint(f'New session! Position is {self.current_position}')
        if first_patch_start is None:
            first_patch_start = np.random.randint(0, 3)
        self.set_current_patch(self.current_patch_num, patch_start=first_patch_start)
        self.total_reward = 0
        self.reward_site_dwell_time = 0
        self.current_reward_site_attempted = False


    def is_position_in_current_patch(self, pos):
        current_patch_bounds = self.current_patch.get_bounds()
        if self.current_reward_site_attempted and pos >= current_patch_bounds[0]:
            return True
        else:
            return (pos >= current_patch_bounds[0] and pos < current_patch_bounds[1])


    def is_agent_in_current_patch(self):
        return self.is_position_in_current_patch(self.current_position)


    def get_reward_site_idx_from_pos(self, pos):
        curr_rws_bounds = self.current_patch.current_reward_site_bounds

        if pos >= curr_rws_bounds[0] and pos < curr_rws_bounds[1]:
            return self.current_patch.current_reward_site_idx
        else:
            return -1


    def get_reward_site_idx_of_current_pos(self):
        return self.get_reward_site_idx_from_pos(self.current_position)

    
    def move_forward(self, dist):
        self.wprint()
        old_position = self.current_position
        self.current_position += dist

        self.wprint(f'Position is {self.current_position}')
        self.wprint(f'Last position was {old_position}')

        immediate_reward = 0

        if self.is_agent_in_current_patch(): # if agent is currently in a patch
            if not self.is_position_in_current_patch(old_position):
                self.wprint('Agent has entered a patch')
                # visual cue given

            current_reward_site_idx = self.get_reward_site_idx_of_current_pos()

            if current_reward_site_idx != -1: # agent is currently at a reward site
                self.reward_site_dwell_time += 1
                # odor cue given
                if self.reward_site_dwell_time >= self.dwell_time_for_reward and not self.current_reward_site_attempted:
                    immediate_reward += self.current_patch.get_reward()
                    self.total_reward += immediate_reward
                    self.current_reward_site_attempted = True
                
                self.wprint(f'Agent is at reward site {current_reward_site_idx}')
                self.wprint(f'Current dwell time is {self.reward_site_dwell_time}')
                self.wprint(f'Reward site attempted: {self.current_reward_site_attempted}')
            elif self.get_reward_site_idx_from_pos(old_position) != -1: # agent is out of a reward site but was just in one
                last_reward_site_idx = self.get_reward_site_idx_from_pos(old_position)
                self.wprint('Agent has left reward site')
                if self.current_reward_site_attempted:
                    self.current_patch.generate_reward_site()
                    self.wprint('New reward site generated')
                    rwsb = self.current_patch.current_reward_site_bounds
                    self.wprint(f'[{rwsb[0]}, {rwsb[1]}]')
                self.reward_site_dwell_time = 0
                self.current_reward_site_attempted = False

        if self.is_position_in_current_patch(old_position) and not self.is_agent_in_current_patch(): # agent was just in a patch but has left
            self.reward_site_dwell_time = 0
            self.current_reward_site_attempted = False
            patch_id = self.generate_next_patch()
            self.wprint(f'Generate patch of type {patch_id}')
            new_patch_start = self.current_patch.current_reward_site_bounds[1] + self.interpatch_len
            self.set_current_patch(patch_id, patch_start=new_patch_start)

        self.wprint(f'Total reward is {self.total_reward}')

        return immediate_reward


    def generate_next_patch(self):
        roll = np.random.rand()
        trans_probs = self.transition_mat[self.last_patch_num]
        cumulative_prob = 0
        next_patch_num = None
        for i in range(len(trans_probs)):
            if roll <= cumulative_prob + trans_probs[i]:
                next_patch_num = i
                break
            cumulative_prob += trans_probs[i]
        return next_patch_num


    def set_current_patch(self, patch_num, patch_start=None):
        self.last_patch_num = self.current_patch_num
        self.current_patch_num = patch_num
        self.current_patch_type = self.patch_types[self.current_patch_num]
        if patch_start is None:
            patch_start = self.current_patch.current_reward_site_bounds[1] + self.interpatch_len
        self.current_patch = self.current_patch_type.generate_patch(patch_start)

        self.wprint(f'New patch created!')
        pb = self.current_patch.get_bounds()
        self.wprint(f'Patch bounds are [{pb[0]}, {pb[1]}]')
        rwsb = self.current_patch.current_reward_site_bounds
        self.wprint(f'Current reward bounds are: [{rwsb[0]}, {rwsb[1]}]')
        # for k in range(len(self.reward_bounds)):
        #     self.wprint(f'[{self.reward_bounds[k][0]}, {self.reward_bounds[k][1]}]')


    def get_observations(self):
        # observations entirely determined by location
        # there are 2 visual cues plus odor cues equal to the number of patchs

        # begin with a noisy vector around all zeros is there is noise in observations
        observations = np.random.normal(loc=0, size=self.obs_size, scale=self.obs_noise_std)

        # [entering_patch_visual_cue, leaving_patch_visual_cue, odor_cue_1, odor_cue_2, ...]
        # if within spatial_buffer_for_visual_cues of start of patch, give `entering_patch_visual_cue`

        if self.is_agent_in_current_patch():
            observations[0] += 1
        
        if self.get_reward_site_idx_of_current_pos() != -1:
            if not self.odor_lesioned:
                observations[1 + self.current_patch.get_odor_num()] += 1
            else:
                observations[1] += 1
        return observations


    def set_verbosity(self, flag):
        self.verbosity = flag


    def wprint(self, inp=''):
        if self.verbosity:
            print(inp)

