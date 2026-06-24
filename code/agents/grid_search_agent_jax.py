import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
import numpy as np
from typing import Tuple


@struct.dataclass
class GridSearchAgentState:
    """Per-timestep mutable state for the JAX grid search agent."""
    dwell_time: jnp.ndarray          # (num_envs,) int32
    odor_site_idx: jnp.ndarray       # (num_envs,) int32
    rewards_in_patch: jnp.ndarray    # (num_envs,) int32
    current_patch_type: jnp.ndarray  # (num_envs,) int32
    last_obs: jnp.ndarray            # (num_envs, obs_size)


def init_agent_state(num_envs: int, obs_size: int) -> GridSearchAgentState:
    return GridSearchAgentState(
        dwell_time=jnp.zeros((num_envs,), dtype=jnp.int32),
        odor_site_idx=jnp.zeros((num_envs,), dtype=jnp.int32),
        rewards_in_patch=jnp.zeros((num_envs,), dtype=jnp.int32),
        current_patch_type=jnp.zeros((num_envs,), dtype=jnp.int32),
        last_obs=jnp.zeros((num_envs, obs_size)),
    )


@partial(jax.jit, static_argnames=['strategy', 'odor_cues_start', 'odor_cues_end',
                                    'patch_cue_idx', 'wait_time_for_reward'])
def grid_search_step(
    state: GridSearchAgentState,
    obs: jnp.ndarray,                # (num_envs, obs_size) — current observation
    n_stops_for_patch: jnp.ndarray,  # (n_patches,) int32
    strategy: str,
    odor_cues_start: int,
    odor_cues_end: int,
    patch_cue_idx: int,
    wait_time_for_reward: int,
) -> Tuple[jnp.ndarray, GridSearchAgentState]:
    """Map (current obs, last obs) → action; update agent state."""
    num_envs = obs.shape[0]

    patch_cue = obs[:, patch_cue_idx]
    last_patch_cue = state.last_obs[:, patch_cue_idx]
    odor_cues = obs[:, odor_cues_start:odor_cues_end]
    last_odor_cues = state.last_obs[:, odor_cues_start:odor_cues_end]
    odor_diff = odor_cues - last_odor_cues

    patch_entered = (patch_cue - last_patch_cue) > 0.5

    # Reset per-patch counters on patch entry
    odor_site_idx = jnp.where(patch_entered, 0, state.odor_site_idx)
    rewards_in_patch = jnp.where(patch_entered, 0, state.rewards_in_patch)

    # Track current patch type whenever any odor cue is active
    odor_active = odor_cues.max(axis=1) > 0.5
    current_patch_type = jnp.where(
        odor_active, odor_cues.argmax(axis=1), state.current_patch_type
    )

    # Per-env odor delta for each env's current patch type
    patch_odor_diff = odor_diff[jnp.arange(num_envs), current_patch_type]
    odor_site_entered = patch_odor_diff > 0.5
    odor_site_exited = patch_odor_diff < -0.5

    # Advance site counter when leaving a site
    odor_site_idx = jnp.where(odor_site_exited, odor_site_idx + 1, odor_site_idx)

    stops_for_patch = n_stops_for_patch[current_patch_type]  # (num_envs,)

    if strategy == 'site_count':
        should_stop = odor_site_entered & (odor_site_idx < stops_for_patch)
    else:  # reward_count
        should_stop = odor_site_entered & (rewards_in_patch < stops_for_patch)

    dwell_time = jnp.where(should_stop, wait_time_for_reward, state.dwell_time)
    action = jnp.where(dwell_time > 0, 0, 1).astype(jnp.int32)
    dwell_time = jnp.where(dwell_time > 0, dwell_time - 1, 0).astype(jnp.int32)

    new_state = state.replace(
        dwell_time=dwell_time,
        odor_site_idx=odor_site_idx,
        rewards_in_patch=rewards_in_patch,
        current_patch_type=current_patch_type,
        last_obs=obs,
    )
    return action, new_state


def advance_policy(
    n_stops_for_patch: np.ndarray,
    stop_ranges: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Increment the policy odometer. Returns (new_stops, search_finished)."""
    if (n_stops_for_patch == stop_ranges[:, 1]).all():
        return n_stops_for_patch.copy(), True
    n_stops = n_stops_for_patch.copy()
    idx = 0
    while idx < len(n_stops):
        if n_stops[idx] < stop_ranges[idx, 1]:
            n_stops[idx] += 1
            break
        else:
            n_stops[idx] = stop_ranges[idx, 0]
            idx += 1
    return n_stops, False
