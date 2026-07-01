"""Single never-ending-patch variant of the treadmill environment.

Copy of treadmill_env_jax.py adapted for curriculum pretraining: the agent is
always inside one patch that never ends, reward sites are regenerated endlessly
ahead of the agent, and the reward probability at each site is a constant
(read from params.reward_prob_prefactors[patch_type]) — no depletion, no
inter-patch travel, no patch-type transitions.

Reuses the PatchState / TreadmillEnvState / TreadmillEnvParams structs and the
`sample_truncated_exp` helper from treadmill_env_jax so the same trajectory
collector and network can be used unchanged.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Tuple, Dict, Any

from environments.treadmill_env_jax import (
    PatchState,
    TreadmillEnvState,
    TreadmillEnvParams,
    treadmill_session_default_params,
    sample_truncated_exp,
)


def TreadmillSinglePatchEnvironment(patch_type: int = 0):
    """Single never-ending patch environment.

    Args:
        patch_type: which odor / patch type the single patch is (0, 1, or 2).
            The constant reward probability is params.reward_prob_prefactors[patch_type].
    """

    step_vals = jnp.array([0.0, 1.0])
    obs_size = 4

    def generate_patch(patch_num, patch_start, reward_params, reward_prob_prefactors):
        return PatchState(
            patch_start=patch_start,
            current_reward_site_idx=-1,
            current_reward_site_start=0,
            current_reward_site_end=0,
            reward_sum=0.0,
            odor_num=patch_num,
            reward_func_param=reward_params[patch_num],
            reward_prob_prefactor=reward_prob_prefactors[patch_num],
            active=True,
        )

    def generate_new_reward_site(key, patch_state, params, first):
        interreward_len = sample_truncated_exp(
            key,
            params.interreward_len_bounds,
            params.interreward_len_decay_rate,
        ).astype(int)

        new_reward_site_start = jnp.where(
            first,
            patch_state.patch_start,
            patch_state.current_reward_site_end,
        ) + interreward_len
        new_reward_site_end = new_reward_site_start + params.reward_site_len

        return patch_state.replace(
            current_reward_site_idx=patch_state.current_reward_site_idx + 1,
            current_reward_site_start=new_reward_site_start,
            current_reward_site_end=new_reward_site_end,
        )

    def get_reward_site_idx_of_current_pos(patch_state, pos):
        return jnp.where(
            jnp.logical_and(pos >= patch_state.current_reward_site_start,
                            pos < patch_state.current_reward_site_end),
            patch_state.current_reward_site_idx,
            -1,
        )

    def is_pos_in_current_patch(state, pos):
        # Single, never-ending patch: in-patch once past the (small) approach offset.
        return pos >= state.current_patch.patch_start

    def get_observations(key, state, params):
        noise = random.normal(key, shape=(obs_size,)) * params.obs_noise_std
        obs = noise

        is_in_patch = is_pos_in_current_patch(state, state.current_position)
        obs = obs.at[0].set(obs[0] + jnp.where(is_in_patch, 1.0, 0.0))

        current_reward_site_idx = get_reward_site_idx_of_current_pos(
            state.current_patch, state.current_position
        )
        is_in_reward_site = (current_reward_site_idx != -1)
        odor_signal = jnp.where(is_in_reward_site, 1.0, 0.0)
        patch_idx = state.current_patch_num + 1  # obs[0] is the visual cue
        obs = obs.at[patch_idx].set(obs[patch_idx] + odor_signal)
        return obs

    @jax.jit
    def reset(key: jnp.ndarray, params: TreadmillEnvParams):
        key, subkey1, subkey2, subkey3 = random.split(key, 4)

        reward_params = params.reward_decay_consts
        reward_prob_prefactors = params.reward_prob_prefactors

        current_patch_num = jnp.array(patch_type)
        current_position = jnp.array(0.0)
        first_patch_start = random.uniform(subkey1, (), minval=1, maxval=3).astype(int)

        patch_state = generate_patch(current_patch_num, first_patch_start,
                                     reward_params, reward_prob_prefactors)
        patch_state = generate_new_reward_site(subkey2, patch_state, params, True)

        state = TreadmillEnvState(
            current_position=current_position,
            current_patch_num=current_patch_num,
            last_patch_num=current_patch_num,
            current_patch=patch_state,
            total_reward=jnp.array(0.0),
            dwell_time=jnp.array(0),
            current_reward_site_attempted=jnp.array(False),
            step_count=jnp.array(0),
            reward_params=reward_params,
            reward_prob_prefactors=reward_prob_prefactors,
            patch_active_transition_probs=jnp.full(
                reward_prob_prefactors.shape,
                params.patch_active_transition_prob_range[0]),
            exp_filtered_reward_rate=jnp.array(0.0),
        )

        obs = get_observations(subkey3, state, params)
        return obs, state

    def move_forward(key, state, dist, params):
        reward_prob = state.current_patch.reward_prob_prefactor   # constant per attempt

        old_position = state.current_position
        new_position = old_position + dist

        old_idx = get_reward_site_idx_of_current_pos(state.current_patch, old_position)
        new_idx = get_reward_site_idx_of_current_pos(state.current_patch, new_position)

        state = state.replace(current_position=new_position)
        is_in_reward_site = (new_idx != -1)

        def in_reward_site(args):
            s, dist, key = args
            new_dwell_time = jnp.where(dist == 0, s.dwell_time + 1, 0)
            s = s.replace(dwell_time=new_dwell_time)

            reward_attempt = (
                (s.dwell_time >= params.dwell_time_for_reward)
                & (~s.current_reward_site_attempted)
            )

            def attempt(key, s):
                r = random.bernoulli(key, reward_prob).astype(jnp.float32)
                patch_state = s.current_patch.replace(
                    reward_sum=s.current_patch.reward_sum + r,
                )
                return s.replace(
                    current_reward_site_attempted=True,
                    current_patch=patch_state,
                    total_reward=s.total_reward + r,
                ), r

            def no_attempt(key, s):
                return s, jnp.array(0.0)

            return lax.cond(reward_attempt, attempt, no_attempt, key, s)

        def out_reward_site(args):
            s, dist, key = args
            was_in_reward_site = (old_idx != -1)
            s = s.replace(dwell_time=0)

            def just_left(s, key):
                # Always generate the next reward site (never leave the patch).
                new_patch_state = generate_new_reward_site(key, s.current_patch, params, False)
                return s.replace(current_reward_site_attempted=False,
                                 current_patch=new_patch_state), jnp.array(0.0)

            return lax.cond(was_in_reward_site, just_left,
                            lambda s, key: (s, jnp.array(0.0)), s, key)

        return lax.cond(is_in_reward_site, in_reward_site, out_reward_site, (state, dist, key))

    @jax.jit
    def step(
        key: jnp.ndarray,
        state: TreadmillEnvState,
        action: jnp.ndarray,
        params: TreadmillEnvParams,
    ) -> Tuple[jnp.ndarray, TreadmillEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:

        key, subkey1, subkey2 = random.split(key, 3)

        forward_movement = step_vals[action]
        state, reward = move_forward(subkey1, state, forward_movement, params)
        state = state.replace(step_count=state.step_count + 1)
        obs = get_observations(subkey2, state, params)

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
            'patch_reward_prob_prefactor': state.current_patch.reward_prob_prefactor,
            'reward': reward,
            'environment_quality': state.reward_params,
        }

        return obs, state, reward, jnp.array(False), info

    return reset, step, get_observations
