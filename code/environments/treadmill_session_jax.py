import jax
import jax.numpy as jnp
from jax import random as jr
from typing import NamedTuple
from gymnax.environments import Environment


class EnvState(NamedTuple):
    position: jnp.ndarray             # scalar int
    patch_type: jnp.ndarray           # scalar int
    current_patch_bounds: jnp.ndarray # shape (2,)
    reward_site_bounds: jnp.ndarray   # shape (2,)
    dwell_time: jnp.ndarray           # scalar int
    reward_attempted: jnp.ndarray     # bool
    total_reward: jnp.ndarray         # scalar float
    reward_in_patch: jnp.ndarray      # scalar float


class MyEnvGymnax(Environment):
    def __init__(self, max_steps=100):
        self.max_steps = max_steps

    def reset(self, rng):
        pos = 0.0
        step_count = 0
        obs = jnp.array([pos])
        env_state = {"pos": pos, "step_count": step_count}
        return env_state, obs

    def step(self, rng, env_state, action):
        pos = env_state["pos"] + action
        step_count = env_state["step_count"] + 1
        done = step_count >= self.max_steps
        reward = -jnp.abs(pos)
        obs = jnp.array([pos])
        new_env_state = {"pos": pos, "step_count": step_count}
        return new_env_state, obs, reward, done, {}

    def action_space(self, seed):
        # Optional: For compatibility
        raise NotImplementedError

    def observation_space(self, seed):
        raise NotImplementedError



def reset(key, batch_size):




def step(state: EnvState, action: jnp.ndarray, key: jax.random.PRNGkey):
    new_position = state.position + action

    agent_in_patch = is_position_in_current_patch(state, new_position)
    at_reward_site = (new_position >= state.reward_site_bounds[0]) & (new_position < state.reward_site_bounds[1])
    entered_patch = (state.position < state.current_patch_bounds[0]) & (new_position >= state.current_patch_bounds[0])
    left_reward_site = (state.position < state.reward_site_bounds[1]) & (new_position >= state.reward_site_bounds[1])

    new_dwell_time = jnp.where(at_reward_site, state.dwell_time + 1, 0)
    reward_ready = (new_dwell_time >= dwell_threshold) & at_reward_site & (~state.reward_attempted) # need to define dwell_threshold
    reward_amount = jnp.where(reward_ready, 1.0 / (1.0 + state.reward_sum), 0.0) # CHANGE THIS

    total_reward = state.total_reward + reward_amount
    reward_sum = jnp.where(agent_in_patch, state.reward_sum + reward_amount, 0)

    reward_attempted = jnp.where(reward_ready, True, jnp.where(at_reward_site, state.reward_attempted, False))

    left_patch = left_reward_site & ~state.reward_attempted

    # Generate new reward site if agent leaves it
    new_rws_bounds = jnp.where(
        left_reward_site & ~left_patch,
        jnp.array([state.reward_site_bounds[1] + interreward_len, state.reward_site_bounds[1] + interreward_len + reward_site_len]),
        state.reward_site_bounds
    )

    new_state = EnvState(
        position=new_position,
        reward_site_bounds=new_rws_bounds,
        dwell_time=new_dwell_time,
        reward_attempted=reward_attempted,
        total_reward=total_reward,
        patch_start=state.patch_start,
        reward_sum=reward_sum,
    )


def is_position_in_current_patch(state: EnvState, pos: jnp.ndarray):
    return jnp.where(
        state.reward_attempted & (pos >= state.current_patch_bounds[0]),
        True,
        (pos >= state.current_patch_bounds[0]) & (pos < state.current_patch_bounds[1]),
    )