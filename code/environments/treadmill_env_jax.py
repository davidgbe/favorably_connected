import jax
import jax.numpy as jnp
from jax import random as jr
from typing import NamedTuple
from gymnax.environments import Environment


@struct.dataclass
class PatchState:
    """JAX version of your Patch class state"""
    patch_start: jnp.ndarray
    current_reward_site_idx: jnp.ndarray
    current_reward_site_start: jnp.ndarray
    current_reward_site_end: jnp.ndarray
    reward_sum: jnp.ndarray
    odor_num: jnp.ndarray
    reward_func_param: jnp.ndarray  # decay constant


@struct.dataclass
class TreadmillEnvState:
    """JAX version of TreadmillSession state"""
    # Position and patch info
    current_position: jnp.ndarray
    current_patch_num: jnp.ndarray
    last_patch_num: jnp.ndarray
    current_patch: PatchState
    
    # Reward tracking
    total_reward: jnp.ndarray
    dwell_time: jnp.ndarray
    current_reward_site_attempted: jnp.ndarray
    
    # Episode info
    step_count: jnp.ndarray


@struct.dataclass
class TreadmillEnvParams:
    """Environment hyperparameters - matches your constants"""
    num_patch_types: int = 3
    obs_size: int = 4
    dwell_time_for_reward: int = 6
    reward_site_len: float = 3.0
    
    # Transition matrix and reward parameters
    transition_mat: jnp.ndarray  # (3, 3)
    reward_decay_consts: jnp.ndarray  # (3,) - [0, 10, 30]
    reward_prob_prefactor: float = 0.8
    
    # Length distributions (truncated exponential)
    interreward_len_bounds: jnp.ndarray = jnp.array([1.0, 6.0])
    interreward_len_decay_rate: float = 0.8
    interpatch_len_bounds: jnp.ndarray = jnp.array([1.0, 12.0])
    interpatch_len_decay_rate: float = 0.1
    
    max_steps: int = 20000


class TreadmillEnvironment:
    """JAX implementation of TreadmillSession"""
    
    def __init__(self, params: Optional[TreadmillEnvParams] = None):
        self.params = params or self.default_params
        
        # Set up action and observation spaces for compatibility
        self.action_space_n = 2  # [0, 1] corresponding to [0, 1] step sizes
        self.observation_space_shape = (self.params.obs_size,)
        self.step_vals = jnp.array([0.0, 1.0])
    
    @property
    def default_params(self) -> TreadmillEnvParams:
        """Default parameters matching your training script constants"""
        # Default uniform transition matrix
        transition_mat = jnp.ones((3, 3)) / 3.0
        
        # Default reward decay constants [0, 10, 30]
        reward_decay_consts = jnp.array([0.0, 10.0, 30.0])
        
        return TreadmillEnvParams(
            num_patch_types=3,
            obs_size=4,
            dwell_time_for_reward=6,
            reward_site_len=3.0,
            obs_noise_std=0.05,
            max_steps=20000,
            odor_lesioned=False,
            reward_decay_consts=reward_decay_consts,
            reward_prob_prefactor=0.8,
            curriculum_style='FIXED',
            reward_decay_range=jnp.array([0.0, 30.0]),
            interreward_len_bounds=jnp.array([1.0, 6.0]),
            interreward_len_decay_rate=0.8,
            interpatch_len_bounds=jnp.array([1.0, 12.0]),
            interpatch_len_decay_rate=0.1,
            transition_mat=transition_mat,
        )
    
    def reset(
        self, 
        key: chex.PRNGKey, 
        params: TreadmillEnvParams,
    ) -> Tuple[jnp.ndarray, TreadmillEnvState]:
        """Reset the environment - equivalent to start_new_session()"""
            
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        
        # Initialize patch number and position
        current_patch_num = random.randint(subkey1, (), 0, params.num_patch_types)
        current_position = jnp.array(0.0)
        
        # Generate first patch starting position (0 to 3, like your code)
        first_patch_start = random.uniform(subkey2, (), minval=0, maxval=3)
        
        # Create initial patch
        patch_state = self._generate_patch(subkey3, current_patch_num, first_patch_start, params)
        patch_state = self._generate_new_reward_site(subkey4, patch_state, params)
        
        state = TreadmillEnvState(
            current_position=current_position,
            current_patch_num=current_patch_num,
            last_patch_num=current_patch_num,  # Initialize to current
            current_patch=patch_state,
            total_reward=jnp.array(0.0),
            dwell_time=jnp.array(0),
            current_reward_site_attempted=jnp.array(False),
            step_count=jnp.array(0),
        )
        
        obs = self._get_observations(state, params)
        return obs, state

    
    def step(
        self,
        key: chex.PRNGKey,
        state: TreadmillEnvState,
        action: jnp.ndarray,
        params: TreadmillEnvParams,
    ) -> Tuple[jnp.ndarray, TreadmillEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:

        key, subkey = random.split(key)
        
        # Convert action to movement amount
        forward_movement = self.step_vals[action]
        
        # Update state through movement
        new_state, reward = self._move_forward(subkey, state, forward_movement, params)
        
        # Update step count and check termination
        new_state = new_state.replace(
            step_count=state.step_count + 1,
        )
        
        # Get new observations
        obs = self._get_observations(new_state, params)
        
        # Create info dict matching your get_info()
        info = {
            'action': forward_movement,
            'current_patch_num': new_state.current_patch_num,
            'current_position': new_state.current_position,
            'current_patch_start': new_state.current_patch.patch_start,
            'reward_bounds': jnp.array([
                new_state.current_patch.current_reward_site_start,
                new_state.current_patch.current_reward_site_end
            ]),
            'agent_in_patch': self._is_pos_in_current_patch(new_state, new_state.current_position),
            'reward_site_idx': self._get_reward_site_idx_of_current_pos(new_state, new_state.current_position),
            'current_reward_site_attempted': new_state.current_reward_site_attempted,
            'patch_reward_param': new_state.current_patch.reward_func_param,
            'reward': reward,
        }
        
        return obs, new_state, reward, jnp.array(False), info
    

    def _move_forward(self, key, state, dist, params):

        # Logical control flow written out as an if/else block
        # if is_in_patch:
        #     if in_reward_site:
        #         if dwell_time > 6 and not reward_site_attempted:
        #         else:
        #     else:
        #         if was_in_reward_site:
        #             if reward_site_attempted:
        #             else:
        #         else:
        # else:
        #     if was_in_patch:
        #     else:

        old_position = state.current_position
        new_position = old_position + dist

        was_in_patch = self._is_pos_in_current_patch(state, old_position)
        is_in_patch = self._is_pos_in_current_patch(state, new_position)

        old_idx = self._get_reward_site_idx_of_current_pos(state, old_position)
        new_idx = self._get_reward_site_idx_of_current_pos(state, new_position)

        def handle_in_patch(args):
            state, old_idx, new_idx, key = args
            is_in_reward_site = (new_idx != -1)

            def in_reward_site(s):
                s = s.replace(reward_site_dwell_time=s.reward_site_dwell_time + 1)
                reward_attempt = (s.reward_site_dwell_time >= params.dwell_time_for_reward) & (~s.current_reward_site_attempted)

                def attempt(s):
                    # Here you'd call _reward_func
                    return s.replace(current_reward_site_attempted=True), jnp.array(1.0)

                def no_attempt(s):
                    return s, jnp.array(0.0)

                return lax.cond(reward_attempt, attempt, no_attempt, s)

            def out_reward_site(s):
                was_in_reward_site = (old_idx != -1)

                def just_left(s):
                    def attempted(s):
                        # Generate new site
                        new_patch_state = self._generate_new_reward_site(key, patch_state, params)
                        return s.replace(current_reward_site_attempted=False, patch_state=new_patch_state), jnp.array(0.0)
                    return lax.cond(s.current_reward_site_attempted, attempted, lambda s: (s, jnp.array(0.0)), s)

                return lax.cond(was_in_reward_site, just_left, lambda s: (s, jnp.array(0.0)), s)

            return lax.cond(is_in_reward_site, in_reward_site, out_reward_site, (state, key))

        def handle_out_patch(args):
            state, was_in_patch = args

            def just_left_patch(s):
                patch_num = _generate_next_patch_type(key, s.current_patch_num, params)

                interpatch_len = sample_truncated_exp(
                    subkey2, 
                    params.interpatch_len_bounds, 
                    params.interpatch_len_decay_rate
                )

                new_patch_start = s.current_patch.current_reward_site_end + interpatch_len
                patch_state = _generate_patch(patch_num, patch_start, + params)

                return s.replace(reward_site_dwell_time=0, current_reward_site_attempted=False), jnp.array(0.0)
            return lax.cond(was_in_patch, just_left_patch, lambda s: (s, jnp.array(0.0)), state)

        return lax.cond(is_in_patch, handle_in_patch, handle_out_patch, (state, old_idx, new_idx, key))


    def _generate_next_patch_type(
        self,
        key: chex.PRNGKey,
        current_patch_num: jnp.ndarray,
        params: TreadmillEnvParams,
    ) -> jnp.ndarray:

        """Generate next patch type based on transition matrix"""
        roll = random.uniform(key)
        trans_probs = params.transition_mat[current_patch_num]
        cumulative_probs = jnp.cumsum(trans_probs)
        
        # Find which patch type to transition to
        next_patch_num = jnp.searchsorted(cumulative_probs, roll)
        return jnp.minimum(next_patch_num, params.num_patch_types - 1)


    def _generate_patch(self, patch_num: jnp.ndarray, patch_start: jnp.ndarray, params: TreadmillEnvParams) -> PatchState:
        """Generate new patch - equivalent to PatchType.generate_patch()"""
        
        # Get reward function parameter based on curriculum style
        reward_func_param = self._get_reward_func_param(patch_num, params)
        
        return PatchState(
            patch_start=patch_start,
            reward_site_len=params.reward_site_len,
            current_reward_site_idx=0,
            current_reward_site_start=reward_site_start,
            current_reward_site_end=reward_site_end,
            reward_sum=jnp.array(0.0),
            odor_num=patch_num,
            reward_func_param=reward_func_param
        )


    def _generate_new_reward_site(
        self,
        key: chex.PRNGKey,
        patch_state: PatchState,
        params: TreadmillEnvParams
    ) -> PatchState:
        """Generate new reward site - equivalent to Patch.generate_reward_site()"""
        # Generate interreward site length
        interreward_len = sample_truncated_exp(
            key, 
            params.interreward_len_bounds, 
            params.interreward_len_decay_rate,
        )
        
        # New reward site starts after current one + interreward distance
        new_reward_site_start = patch_state.current_reward_site_end + interreward_len
        new_reward_site_end = new_reward_site_start + patch_state.reward_site_len
        
        return patch_state.replace(
            current_reward_site_idx=patch_state.current_reward_site_idx + 1,
            current_reward_site_start=new_reward_site_start,
            current_reward_site_end=new_reward_site_end
        )


    def _reward_func(self, key: chex.PRNGKey, reward_sum: jnp.ndarray, decay_const: jnp.ndarray) -> jnp.ndarray:
        """Stochastic reward function with exponential decay"""
        # reward_prob = prefactor * exp(-reward_sum / decay_const)
        # return bernoulli sample
        pass
    
    def _get_observations(self, state: TreadmillEnvState, params: TreadmillEnvParams) -> jnp.ndarray:
        """Get observations - equivalent to get_observations()"""
        # [visual_cue, odor_cue_0, odor_cue_1, odor_cue_2] + noise
        pass

    
def sample_truncated_exp(
    key: chex.PRNGKey, 
    bounds: jnp.ndarray, 
    decay_rate: float
) -> jnp.ndarray:
    """Sample from truncated exponential distribution
    
    Args:
        key: Random key
        bounds: [min_val, max_val] bounds for truncation
        decay_rate: Exponential decay rate parameter
    
    Returns:
        Sample from truncated exponential
    """
    min_val, max_val = bounds[0], bounds[1]
    
    # Sample from uniform [0, 1)
    u = random.uniform(key)
    
    # Convert to truncated exponential using inverse CDF
    # F(x) = (1 - exp(-decay_rate * x)) / (1 - exp(-decay_rate * max_val))
    # for x in [0, max_val], then shift by min_val
    
    exp_max = jnp.exp(-decay_rate * (max_val - min_val))
    sample = -jnp.log(1 - u * (1 - exp_max)) / decay_rate
    
    return min_val + sample