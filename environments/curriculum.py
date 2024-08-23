import numpy as np
import gymnasium as gym

class Curriculum():

    def __init__(self, curriculum_step_starts, curriculum_step_env_funcs):
        self.curriculum_step_starts = curriculum_step_starts
        self.curriculum_step_env_funcs = curriculum_step_env_funcs
        self.step_count = 0
        self.current_envs = None
        self.call_count = 0

    def get_envs_for_step(self, env_seeds):
        if self.current_envs is None:
            self.current_envs = gym.vector.AsyncVectorEnv([self.curriculum_step_env_funcs[0](i) for i in env_seeds])
            self.step_count += 1
        elif self.step_count < len(self.curriculum_step_starts) and self.call_count == self.curriculum_step_starts[self.step_count]:
            self.current_envs = gym.vector.AsyncVectorEnv([self.curriculum_step_env_funcs[self.step_count](i + self.step_count) for i in env_seeds])
            if self.step_count < (len(self.curriculum_step_starts) - 1):
                self.step_count += 1
        self.call_count += 1
        return self.current_envs
