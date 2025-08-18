import jax
import jax.numpy as jnp
from jax import random, lax
import chex
from flax import struct
from typing import Tuple, Dict, Any, Optional
from functools import partial


@struct.dataclass
class PatchState:
    """JAX version of your Patch class state"""
    patch_start: float
    current_reward_site_idx: int
    current_reward_site_start: int
    current_reward_site_end: int
    reward_sum: float
    odor_num: int
    reward_func_param: float  # decay constant


@struct.dataclass
class TreadmillEnvState:
    """JAX version of TreadmillSession state"""
    # Position and patch info
    current_position: int
    current_patch_num: int
    last_patch_num: int
    current_patch: PatchState
    
    # Reward tracking
    total_reward: float
    dwell_time: int
    current_reward_site_attempted: bool
    
    # Episode info
    step_count: int


@struct.dataclass
class TreadmillEnvParams:

    # Transition matrix and reward parameters
    transition_mat: jnp.ndarray  # (3, 3)
    reward_decay_consts: jnp.ndarray  # (3,) - [0, 10, 30]
    reward_prob_prefactor: float
    reward_decay_range: jnp.ndarray # (2,) - [0, 30]
    
    # Length distributions (truncated exponential)
    interreward_len_bounds: jnp.ndarray
    interreward_len_decay_rate: float
    interpatch_len_bounds: jnp.ndarray
    interpatch_len_decay_rate: float

    """Environment hyperparameters - matches your constants"""
    num_patch_types: int = 3
    obs_size: int = 4
    action_size: int = 2
    dwell_time_for_reward: int = 6
    reward_site_len: int = 3
    obs_noise_std: float = 0.05
    
    max_steps: int = 20000
    # curriculum_style: str = 'FIXED'


def treadmill_session_default_params() -> TreadmillEnvParams:
    """Default parameters matching your training script constants"""
    # Default uniform transition matrix
    transition_mat = jnp.ones((3, 3)) / 3.0
    
    # Default reward decay constants [0, 10, 30]
    reward_decay_consts = jnp.array([0.0, 10.0, 30.0])
    
    return TreadmillEnvParams(
        num_patch_types=3,
        obs_size=4,
        dwell_time_for_reward=6,
        reward_site_len=3,
        obs_noise_std=0.05,
        max_steps=20000,
        reward_decay_consts=reward_decay_consts,
        reward_prob_prefactor=0.8,
        reward_decay_range=jnp.array([0.0, 30.0]),
        interreward_len_bounds=jnp.array([1.0, 6.0]),
        interreward_len_decay_rate=0.8,
        interpatch_len_bounds=jnp.array([1.0, 12.0]),
        interpatch_len_decay_rate=0.1,
        transition_mat=transition_mat,
    )


def TreadmillEnvironment():
    """JAX implementation of TreadmillSession"""
    
    step_vals = jnp.array([0.0, 1.0])
    obs_size = 4
    

    @jax.jit
    def reset(
        key: chex.PRNGKey, 
        params: TreadmillEnvParams,
    ) -> Tuple[jnp.ndarray, TreadmillEnvState]:
        """Reset the environment - equivalent to start_new_session()"""
            
        key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
        
        # Initialize patch number and position
        current_patch_num = random.randint(subkey1, (), 0, params.num_patch_types)
        current_position = jnp.array(0.0)
        
        # Generate first patch starting position (0 to 3, like your code)
        first_patch_start = random.uniform(subkey2, (), minval=1, maxval=3).astype(int)
        
        # Create initial patch
        patch_state = generate_patch(current_patch_num, first_patch_start, params)
        patch_state = generate_new_reward_site(subkey3, patch_state, params, True)
        
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
        
        obs = get_observations(subkey4, state, params)
        return obs, state

    
    @jax.jit
    def step(
        key: chex.PRNGKey,
        state: TreadmillEnvState,
        action: jnp.ndarray,
        params: TreadmillEnvParams,
    ) -> Tuple[jnp.ndarray, TreadmillEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:

        key, subkey1, subkey2 = random.split(key, 3)
        
        # Convert action to movement amount
        forward_movement = step_vals[action]
        
        # Update state through movement
        state, reward = move_forward(subkey1, state, forward_movement, params)
        
        # Update step count and check termination
        state = state.replace(
            step_count=state.step_count + 1,
        )
        
        # Get new observations
        obs = get_observations(subkey2, state, params)
        
        # Create info dict matching your get_info()
        info = {
            'action': forward_movement,
            'current_patch_num': state.current_patch_num,
            'current_position': state.current_position,
            'current_patch_start': state.current_patch.patch_start,
            'reward_bounds': jnp.array([
                state.current_patch.current_reward_site_start,
                state.current_patch.current_reward_site_end
            ]),
            'agent_in_patch': is_pos_in_current_patch(state, state.current_position),
            'reward_site_idx': get_reward_site_idx_of_current_pos(state.current_patch, state.current_position),
            'current_reward_site_attempted': state.current_reward_site_attempted,
            'patch_reward_param': state.current_patch.reward_func_param,
            'reward': reward,
        }
        
        return obs, state, reward, jnp.array(False), info


    def move_forward(key, state, dist, params):
        old_position = state.current_position
        new_position = old_position + dist

        was_in_patch = is_pos_in_current_patch(state, old_position)
        is_in_patch = is_pos_in_current_patch(state, new_position)

        old_idx = get_reward_site_idx_of_current_pos(state.current_patch, old_position)
        new_idx = get_reward_site_idx_of_current_pos(state.current_patch, new_position)

        state = state.replace(current_position=new_position)

        def handle_in_patch(args):
            state, was_in_patch, old_idx, new_idx, key = args
            is_in_reward_site = (new_idx != -1)

            def in_reward_site(args):
                s, old_idx, key = args
                s = s.replace(dwell_time=s.dwell_time + 1)
                reward_attempt = (s.dwell_time >= params.dwell_time_for_reward) & (~s.current_reward_site_attempted)

                def attempt(key, s):
                    r = reward_func(
                        key,
                        s.current_patch.reward_sum,
                        s.current_patch.reward_func_param,
                        params
                    )

                    patch_state = s.current_patch.replace(reward_sum=s.current_patch.reward_sum + r)
                    
                    return s.replace(
                        current_reward_site_attempted=True,
                        current_patch=patch_state,
                        total_reward=s.total_reward + r,
                    ), r

                def no_attempt(key, s):
                    return s, jnp.array(0.0)

                return lax.cond(reward_attempt, attempt, no_attempt, key, s)

            def out_reward_site(args):
                s, old_idx, key = args
                was_in_reward_site = (old_idx != -1)
                s = s.replace(dwell_time=0)

                def just_left(s, key):
                    def attempted(s, key):
                        # Generate new site
                        new_patch_state = generate_new_reward_site(key, s.current_patch, params, False)
                        return s.replace(current_reward_site_attempted=False, current_patch=new_patch_state), jnp.array(0.0)
                    
                    return lax.cond(
                        s.current_reward_site_attempted,
                        attempted,
                        lambda s, key: (s, jnp.array(0.0)),
                        s,
                        key
                    )

                return lax.cond(was_in_reward_site, just_left, lambda s, key: (s, jnp.array(0.0)), s, key)

            return lax.cond(is_in_reward_site, in_reward_site, out_reward_site, (state, old_idx, key))

        def handle_out_patch(args):
            state, was_in_patch, old_idx, new_idx, key = args

            def just_left_patch(key, s):
                key, subkey1, subkey2 = random.split(key, 3)
                patch_num = generate_next_patch_type(subkey1, s.current_patch_num, params)

                interpatch_len = sample_truncated_exp(
                    key, 
                    params.interpatch_len_bounds, 
                    params.interpatch_len_decay_rate
                ).astype(int)

                new_patch_start = s.current_patch.current_reward_site_end + interpatch_len

                patch_state = generate_patch(patch_num, new_patch_start, params)
                patch_state = generate_new_reward_site(subkey2, patch_state, params, True)

                return s.replace(
                    dwell_time=0,
                    current_reward_site_attempted=False,
                    current_patch=patch_state,
                    current_patch_num=patch_num,
                    last_patch_num=s.current_patch_num
                ), jnp.array(0.0)
            
            return lax.cond(was_in_patch, just_left_patch, lambda key, s: (s, jnp.array(0.0)), key, state)

        return lax.cond(is_in_patch, handle_in_patch, handle_out_patch, (state, was_in_patch, old_idx, new_idx, key))


    def generate_next_patch_type(
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


    def generate_patch(patch_num: jnp.ndarray, patch_start: jnp.ndarray, params: TreadmillEnvParams) -> PatchState:
        """Generate new patch - equivalent to PatchType.generate_patch()"""
        
        # Get reward function parameter
        reward_func_param = params.reward_decay_consts[patch_num]
        
        return PatchState(
            patch_start=patch_start,
            current_reward_site_idx=-1,
            current_reward_site_start=0,
            current_reward_site_end=0,
            reward_sum=0.0,
            odor_num=patch_num,
            reward_func_param=reward_func_param
        )


    def generate_new_reward_site(
        key: chex.PRNGKey,
        patch_state: PatchState,
        params: TreadmillEnvParams,
        first: bool,
    ) -> PatchState:
        """Generate new reward site - equivalent to Patch.generate_reward_site()"""
        # Generate interreward site length
        interreward_len = sample_truncated_exp(
            key, 
            params.interreward_len_bounds, 
            params.interreward_len_decay_rate,
        ).astype(int)
        
        # New reward site starts after current one + interreward distance
        new_reward_site_start = jnp.where(
            first,
            patch_state.patch_start,
            patch_state.current_reward_site_end,
        ) + interreward_len

        new_reward_site_end = new_reward_site_start + params.reward_site_len
        
        return patch_state.replace(
            current_reward_site_idx=patch_state.current_reward_site_idx + 1,
            current_reward_site_start=new_reward_site_start,
            current_reward_site_end=new_reward_site_end
        )


    def reward_func(key: chex.PRNGKey, reward_sum: jnp.ndarray, decay_const: jnp.ndarray, params: TreadmillEnvParams) -> jnp.ndarray:
        """Stochastic reward function with exponential decay"""
        # Avoid division by zero when decay_const = 0
        reward_prob = jnp.where(
            decay_const == 0,
            0,
            params.reward_prob_prefactor * jnp.exp(-reward_sum / jnp.maximum(decay_const, 1e-8))
        )
        return random.bernoulli(key, reward_prob).astype(jnp.float32)


    def get_observations(key: chex.PRNGKey, state: TreadmillEnvState, params: TreadmillEnvParams) -> jnp.ndarray:
        """Get observations - equivalent to get_observations()
        
        Returns:
            jnp.ndarray of shape (4,): [visual_cue, odor_cue_0, odor_cue_1, odor_cue_2] + noise
        """
        # Start with noise baseline
        noise = random.normal(key, shape=(obs_size,)) * params.obs_noise_std
        
        # Initialize observation with noise
        obs = noise
        
        # Check if agent is in current patch
        is_in_patch = is_pos_in_current_patch(state, state.current_position)
        
        # Visual cue: add 1.0 if agent is in patch (regardless of reward site)
        obs = obs.at[0].set(obs[0] + jnp.where(is_in_patch, 1.0, 0.0))
        
        # Check if agent is in a reward site
        current_reward_site_idx = get_reward_site_idx_of_current_pos(
            state.current_patch, state.current_position
        )
        is_in_reward_site = (current_reward_site_idx != -1)
        
        # Odor cue: add 1.0 to the appropriate odor channel if in reward site
        # Only show odor when in reward site (more specific than just being in patch)
        odor_signal = jnp.where(is_in_reward_site, 1.0, 0.0)
        patch_idx = state.current_patch_num + 1  # Offset by 1 since obs[0] is visual
        obs = obs.at[patch_idx].set(obs[patch_idx] + odor_signal)
        
        return obs


    def is_pos_in_current_patch(
        state: TreadmillEnvState,
        pos: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns True if `pos` lies within the current patch bounds.
        """
        return jnp.where(
            state.current_reward_site_attempted,
            pos > state.current_patch.patch_start,
            (pos >= state.current_patch.patch_start) & (pos < state.current_patch.current_reward_site_end),
        )


    def get_reward_site_idx_of_current_pos(
        patch_state: PatchState,
        pos: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Returns the index of the reward site the position is in,
        or -1 if not in any reward site.
        """
        return jnp.where(
            jnp.logical_and(pos >= patch_state.current_reward_site_start, pos < patch_state.current_reward_site_end),
            patch_state.current_reward_site_idx,
            -1,
        )
    
    return reset, step, get_observations
    
    
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