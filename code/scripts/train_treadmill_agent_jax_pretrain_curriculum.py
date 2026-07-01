"""Curriculum *pretraining* of the actor network across a sequence of tasks.

Stages are listed (and ordered) in a JSON config; each stage warm-starts from
the previous stage's parameters.  Three task types are supported:

  'integration'  - supervised: integrate input-1 (uniform[0,1] pulses arriving
                   on a Poisson schedule) while a gate input (input-2) is on;
                   the target resets to 0 whenever the gate drops below 0.5.
                   The gate is on/off for Poisson-distributed runs.  Only the
                   actor GRU (+ a dedicated scalar readout head) is trained.

  'single_patch' - A2C in a single, never-ending patch with a constant
                   per-site reward probability (no depletion, no travel).

  'foraging'     - A2C in the full treadmill foraging environment.

Inputs are always 7-D ([obs(4), prev_action(2), prev_reward(1)]); the two
integration inputs occupy channels 0 and 1 so the same network/params flow
through every stage unchanged.
"""

import sys
from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional
from functools import partial

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import jax
import jax.numpy as jnp

if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

from jax import random, lax
from flax import serialization
from flax.core import freeze, unfreeze
from flax.training import checkpoints
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import trange
import argparse
import pickle
import json

from aux_funcs import zero_pad
from agents.a2c_rnn_flax import init_network_and_params, A2CRNNFlax
from environments.components.train_state import create_train_state, init_opt
from environments.components.treadmill_trajectory import collect_trajectory
from environments.components.treadmill_trajectory_single_patch import collect_trajectory_single_patch
from environments.treadmill_env_jax import (
    TreadmillEnvironment,
    treadmill_session_default_params,
)
from environments.treadmill_env_single_patch_jax import TreadmillSinglePatchEnvironment
from train_treadmill_agent_jax import (
    N_UPDATES_PER_SESSION,
    N_STEPS_PER_UPDATE,
    run_session_updates_with_metrics,
    compute_n_step_returns,
    reward_param_style_str_to_int,
    reward_func_type_str_to_int,
)

INPUT_DIM = 7  # obs(4) + prev_action(2) + prev_reward(1)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class CurriculumStep:
    task: str = 'foraging'   # 'integration' | 'single_patch' | 'foraging'
    n_sessions: int = 100
    output_save_start: int = 0
    output_save_step: int = 50
    output_save_end: Optional[int] = None
    learning_rate: float = 0.0001

    # --- foraging / single_patch (env) params ---
    dwell_time_for_reward: int = 3
    reward_site_len: int = 3
    fixed_patches: List = field(default_factory=lambda: [0, 0, 0])
    reward_param_style: str = 'fixed'
    reward_func_type: str = 'exp'
    reward_decay_consts: List = field(default_factory=lambda: [0.0, 10.0, 30.0])
    reward_prob_prefactors: List = field(default_factory=lambda: [0.8, 0.8, 0.8])
    reward_decay_range: List = field(default_factory=lambda: [0.0, 40.0])
    reward_prob_range: List = field(default_factory=lambda: [0.05, 0.95])
    patch_active_transition_prob_range: List = field(default_factory=lambda: [0.9, 0.9])
    interreward_len_bounds: List = field(default_factory=lambda: [1.0, 6.0])
    interreward_len_decay_rate: float = 0.8
    interpatch_len_bounds: List = field(default_factory=lambda: [1.0, 12.0])
    interpatch_len_decay_rate: float = 0.1
    global_reward_weight: float = 1.0

    # --- single_patch params ---
    single_patch_type: int = 0
    single_patch_reward_prob: float = 0.5

    # --- integration params ---
    integ_gate_lambda: float = 20.0    # Poisson mean for gate on/off run lengths
    integ_input_lambda: float = 3.0    # Poisson mean for input-1 inter-arrival
    integ_batch_size: Optional[int] = None  # defaults to config.num_envs


@dataclass
class CurriculumConfig:
    exp_name: str = ''
    num_envs: int = 128
    patch_types_per_env: int = 3
    obs_size: int = 4
    action_size: int = 2
    hidden_size: int = 64
    critic_weight: float = 0.05
    entropy_weight: float = 0.0025
    env_prediction_weight: float = 0.0
    activity_norm_weight: float = 0.001
    pred_obs_weight: float = 0.0
    gamma: float = 0.999
    input_noise_std: float = 0.01
    unit_noise_std: float = 0.01
    rnn_type: str = 'GRU'
    init_scale: float = 1.0
    seed: int = 0
    # Reinitialize input weights (RNN input projections) and the actor logits
    # head at every curriculum transition (keeps recurrent/critic/aux weights).
    reinit_io_at_transition: bool = True
    curriculum: List[CurriculumStep] = field(default_factory=list)


def load_curriculum_config(filepath: str) -> CurriculumConfig:
    with open(filepath, 'r') as f:
        d = json.load(f)
    steps = [CurriculumStep(**s) for s in d.pop('curriculum')]
    return CurriculumConfig(curriculum=steps, **d)


def build_env_params(step: CurriculumStep):
    base = treadmill_session_default_params()
    return base.replace(
        reward_param_style=reward_param_style_str_to_int(step.reward_param_style),
        reward_func_type=reward_func_type_str_to_int(step.reward_func_type),
        fixed_patches=jnp.array(step.fixed_patches),
        reward_decay_consts=jnp.array(step.reward_decay_consts),
        reward_prob_prefactors=jnp.array(step.reward_prob_prefactors),
        reward_decay_range=jnp.array(step.reward_decay_range),
        reward_prob_range=jnp.array(step.reward_prob_range),
        patch_active_transition_prob_range=jnp.array(step.patch_active_transition_prob_range),
        interreward_len_bounds=jnp.array(step.interreward_len_bounds),
        interreward_len_decay_rate=step.interreward_len_decay_rate,
        interpatch_len_bounds=jnp.array(step.interpatch_len_bounds),
        interpatch_len_decay_rate=step.interpatch_len_decay_rate,
        dwell_time_for_reward=step.dwell_time_for_reward,
        reward_site_len=step.reward_site_len,
    )


def build_single_patch_env_params(step: CurriculumStep):
    """Env params for the single-patch stage: constant reward prob at the chosen
    patch type, normal odor-site spacing/dwell."""
    base = treadmill_session_default_params()
    prefactors = [0.0, 0.0, 0.0]
    prefactors[step.single_patch_type] = step.single_patch_reward_prob
    return base.replace(
        reward_prob_prefactors=jnp.array(prefactors),
        interreward_len_bounds=jnp.array(step.interreward_len_bounds),
        interreward_len_decay_rate=step.interreward_len_decay_rate,
        dwell_time_for_reward=step.dwell_time_for_reward,
        reward_site_len=step.reward_site_len,
    )


def should_save(session_in_step: int, n_sessions: int, save_start: int,
                save_step: int, save_end: Optional[int]) -> bool:
    if save_start == -1:
        return session_in_step == n_sessions - 1
    end = save_end if save_end is not None else n_sessions
    return (save_start <= session_in_step < end and
            (session_in_step - save_start) % save_step == 0)


# ----------------------------------------------------------------------------
# Integration task (supervised)
# ----------------------------------------------------------------------------
def generate_integration_batch(rng, batch, T, gate_lambda, input_lambda):
    """Generate one batch of integration sequences.

    Returns:
        inputs:  (batch, T, INPUT_DIM)  channels 0=input-1 (pulses), 1=gate
        targets: (batch, T)             running integral of input-1 while gated,
                                        reset to 0 whenever the gate is off.
    """
    inp = np.zeros((batch, T, INPUT_DIM), dtype=np.float32)
    inp1 = np.zeros((batch, T), dtype=np.float32)
    gate = np.zeros((batch, T), dtype=np.float32)

    for b in range(batch):
        # Gate: alternating on/off runs, each length ~ Poisson(gate_lambda) (>=1)
        t = 0
        state = int(rng.integers(0, 2))
        while t < T:
            dur = max(1, int(rng.poisson(gate_lambda)))
            gate[b, t:t + dur] = state
            t += dur
            state = 1 - state
        # Input-1 pulses: uniform[0,1] at Poisson(input_lambda) inter-arrivals
        t = int(rng.poisson(input_lambda))
        while t < T:
            inp1[b, t] = rng.random()
            t += max(1, int(rng.poisson(input_lambda)))

    # Target: cumulative sum of input-1 over each contiguous gate-on run.
    tgt = np.zeros((batch, T), dtype=np.float32)
    acc = np.zeros(batch, dtype=np.float32)
    on = gate > 0.5
    for ti in range(T):
        acc = np.where(on[:, ti], acc + inp1[:, ti], 0.0)
        tgt[:, ti] = acc

    inp[:, :, 0] = inp1
    inp[:, :, 1] = gate
    return jnp.asarray(inp), jnp.asarray(tgt)


def add_integration_head(network, params, rng_key, hidden_size):
    """Graft the (uninitialised) integration_prediction head params into `params`.

    __call__ never touches that head, so it is absent from the params returned by
    init_network_and_params; we initialise it via the `integrate` method.
    """
    dummy_x = jnp.zeros((1, INPUT_DIM))
    dummy_h = jnp.zeros((1, hidden_size))
    integ_vars = network.init(
        {'params': rng_key, 'noise': rng_key},
        dummy_x, dummy_h, method=A2CRNNFlax.integrate,
    )
    p = unfreeze(params)
    p['params']['integration_prediction'] = unfreeze(integ_vars)['params']['integration_prediction']
    return freeze(p)


def reinitialize_io_weights(config, params, rng_key):
    """Reinitialize the network's input weights (RNN input projections for both
    the actor and critic) and the actor logits output head, leaving recurrent,
    critic-value, auxiliary, and integration weights intact.

    Used at curriculum transitions so the network relearns the input encoding
    and action mapping for the new task while keeping the recurrent dynamics it
    has acquired.
    """
    _, fresh = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        unit_noise_std=config.unit_noise_std,
        rng_key=rng_key,
        init_scale=config.init_scale,
    )
    input_keys = ['in', 'ir', 'iz'] if config.rnn_type == 'GRU' else ['input_projection']
    p = unfreeze(params)
    fp = unfreeze(fresh)
    for cell in ['rnn_actor', 'rnn_critic']:
        for k in input_keys:
            p['params'][cell][k] = fp['params'][cell][k]
    p['params']['actor'] = fp['params']['actor']   # output head -> action logits
    return freeze(p)


@partial(jax.jit, static_argnames=['hidden_size', 'obs_size', 'action_size', 'rnn_type', 'unit_noise_std'])
def integration_train_step(params, opt_state, inputs, targets, rng_key, learning_rate,
                           hidden_size, obs_size, action_size, rnn_type, unit_noise_std):
    network = A2CRNNFlax(
        action_size=action_size, obs_size=obs_size, hidden_size=hidden_size,
        rnn_type=rnn_type, unit_noise_std=unit_noise_std,
    )
    batch = inputs.shape[0]
    inputs_tm = jnp.swapaxes(inputs, 0, 1)   # (T, batch, INPUT_DIM)

    def loss_fn(params):
        h0 = jnp.zeros((batch, hidden_size))

        def scan_t(carry, x_t):
            h, key = carry
            key, noise_key = random.split(key)
            pred, h = network.apply(
                params, x_t, h, method=A2CRNNFlax.integrate,
                rngs={'noise': noise_key},
            )
            return (h, key), pred[:, 0]

        (_, _), preds = lax.scan(scan_t, (h0, rng_key), inputs_tm)   # preds (T, batch)
        preds = jnp.swapaxes(preds, 0, 1)                            # (batch, T)
        return jnp.mean((preds - targets) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.apply_if_finite(optax.adam(learning_rate), max_consecutive_errors=100),
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def run_integration_step(config, step, step_idx, train_state, rng, session_abs,
                         all_session_metrics, step_weights_dir, step_rewards_dir):
    batch = step.integ_batch_size or config.num_envs
    T = N_STEPS_PER_UPDATE

    for session_in_step in trange(step.n_sessions, desc=f'Step {step_idx} (integration)'):
        session_losses = []
        for _ in range(N_UPDATES_PER_SESSION):
            inputs, targets = generate_integration_batch(
                rng, batch, T, step.integ_gate_lambda, step.integ_input_lambda,
            )
            key = train_state.rng_key
            key, sub = random.split(key)
            params, opt_state, loss = integration_train_step(
                train_state.params, train_state.opt_state, inputs, targets, sub,
                step.learning_rate, config.hidden_size, config.obs_size,
                config.action_size, config.rnn_type, config.unit_noise_std,
            )
            train_state = train_state.replace(params=params, opt_state=opt_state, rng_key=key)
            session_losses.append(float(loss))

        session_mean_loss = float(np.mean(session_losses))
        all_session_metrics.append(-session_mean_loss)  # negate so "higher is better" on the shared plot
        print(f'Step {step_idx} sess {session_in_step} (abs {session_abs}): '
              f'integration MSE={session_mean_loss:.5f}')

        sn = zero_pad(session_in_step, 6)
        with open(step_rewards_dir / sn, 'ab') as f:
            np.save(f, np.array(session_losses))

        if should_save(session_in_step, step.n_sessions, step.output_save_start,
                       step.output_save_step, step.output_save_end):
            checkpoints.save_checkpoint(
                ckpt_dir=str(step_weights_dir),
                target={'params': train_state.params},
                step=session_abs, overwrite=False, keep=float('inf'),
            )
            print(f'  -> Saved weights to {step_weights_dir}/checkpoint_{session_abs}')

        session_abs += 1

    return train_state, session_abs


# ----------------------------------------------------------------------------
# Single-patch task (A2C on the never-ending patch)
# ----------------------------------------------------------------------------
def _a2c_losses(trajectory, gamma, critic_weight, entropy_weight,
                env_prediction_weight, global_reward_weight, activity_norm_weight,
                pred_obs_weight, obs_size):
    """Shared A2C loss math (mirrors compute_a2c_loss in train_treadmill_agent_jax)."""
    logits = trajectory.logits
    values = trajectory.values

    returns = jax.vmap(compute_n_step_returns, (0, None, 0))(
        trajectory.rewards, gamma, lax.stop_gradient(values[:, -1])
    )
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    log_probs = jax.nn.log_softmax(logits)
    chosen_log_probs = jnp.take_along_axis(
        log_probs, lax.stop_gradient(trajectory.actions[..., None]), axis=-1
    ).squeeze(-1)

    actor_loss = -jnp.mean(chosen_log_probs * lax.stop_gradient(advantages))
    critic_loss = jnp.mean((values - lax.stop_gradient(returns)) ** 2)

    env_quality_prediction_loss = jnp.mean(
        (trajectory.pred_environment_quality - lax.stop_gradient(trajectory.environment_quality)) ** 2
    )
    global_reward_rate_loss = jnp.mean(
        (trajectory.pred_reward_rate.squeeze() - lax.stop_gradient(trajectory.exp_filtered_reward_rate)) ** 2
    )
    obs_plus_rewards = jnp.concatenate(
        (trajectory.observations[..., :obs_size], trajectory.rewards[..., None]), axis=2
    )
    pred_obs_loss = jnp.mean(
        (trajectory.pred_obs[:, :-1, :] - lax.stop_gradient(obs_plus_rewards[:, 1:, :])) ** 2
    )

    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    entropy_loss = -jnp.mean(entropy)

    activity_norm = (
        jnp.linalg.norm(trajectory.actor_hidden, axis=2).mean()
        + jnp.linalg.norm(trajectory.critic_hidden, axis=2).mean()
    )

    total_loss = (
        actor_loss
        + critic_weight * critic_loss
        + entropy_weight * entropy_loss
        + activity_norm_weight * activity_norm
        + env_prediction_weight * env_quality_prediction_loss
        + pred_obs_weight * pred_obs_loss
        + global_reward_weight * global_reward_rate_loss
    )
    metrics = {
        'total_loss': total_loss,
        'mean_reward': jnp.mean(trajectory.rewards),
    }
    return total_loss, metrics


def compute_a2c_loss_sp(params, train_state, env_states, env_params,
                        gamma, critic_weight, entropy_weight, env_prediction_weight,
                        global_reward_weight, activity_norm_weight, pred_obs_weight,
                        input_noise_std, hidden_size, unit_noise_std, rnn_type, obs_size,
                        patch_type):
    trajectory, final_train_state, final_env_states = collect_trajectory_single_patch(
        train_state=train_state.replace(params=params),
        env_states=env_states,
        env_params=env_params,
        input_noise_std=input_noise_std,
        unit_noise_std=unit_noise_std,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        obs_size=obs_size,
        n_steps=N_STEPS_PER_UPDATE,
        patch_type=patch_type,
    )
    total_loss, metrics = _a2c_losses(
        trajectory, gamma, critic_weight, entropy_weight, env_prediction_weight,
        global_reward_weight, activity_norm_weight, pred_obs_weight, obs_size,
    )
    return total_loss, (metrics, lax.stop_gradient(final_train_state),
                        lax.stop_gradient(final_env_states))


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'obs_size', 'patch_type'])
def train_step_sp(train_state, env_states, env_params, gamma, critic_weight,
                  entropy_weight, env_prediction_weight, global_reward_weight,
                  activity_norm_weight, pred_obs_weight, input_noise_std,
                  hidden_size, unit_noise_std, rnn_type, obs_size, patch_type):
    grad_fn = jax.grad(compute_a2c_loss_sp, has_aux=True)
    grads, (metrics, final_train_state, final_env_states) = grad_fn(
        train_state.params, train_state, env_states, env_params,
        gamma, critic_weight, entropy_weight, env_prediction_weight,
        global_reward_weight, activity_norm_weight, pred_obs_weight,
        input_noise_std, hidden_size, unit_noise_std, rnn_type, obs_size, patch_type,
    )
    metrics['grad_norm'] = optax.global_norm(grads)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.apply_if_finite(optax.adam(train_state.learning_rate), max_consecutive_errors=100),
    )
    updates, new_opt_state = optimizer.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    final_train_state = final_train_state.replace(params=new_params, opt_state=new_opt_state)
    return final_train_state, final_env_states, metrics


def run_single_patch_step(config, step, step_idx, train_state, env_params, reset_fn,
                          session_abs, all_session_metrics, step_traj_dir,
                          step_weights_dir, step_rewards_dir):
    patch_type = int(step.single_patch_type)
    sp_reset_fn, _, _ = TreadmillSinglePatchEnvironment(patch_type)

    for session_in_step in trange(step.n_sessions, desc=f'Step {step_idx} (single_patch)'):
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(sp_reset_fn, in_axes=(0, None))(reset_keys, env_params)

        train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            critic_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            prev_action=jnp.zeros((config.num_envs,), dtype=jnp.int32),
            prev_reward=jnp.zeros((config.num_envs,)),
            prev_obs=obs,
        )

        update_rewards = []
        update_grad_norms = []
        for _ in range(N_UPDATES_PER_SESSION):
            train_state, env_states, metrics = train_step_sp(
                train_state, env_states, env_params,
                config.gamma, config.critic_weight, config.entropy_weight,
                config.env_prediction_weight, step.global_reward_weight,
                config.activity_norm_weight, config.pred_obs_weight,
                config.input_noise_std, config.hidden_size, config.unit_noise_std,
                config.rnn_type, config.obs_size, patch_type,
            )
            update_rewards.append(float(metrics['mean_reward']))
            update_grad_norms.append(float(metrics['grad_norm']))

        session_mean_reward = float(np.mean(update_rewards))
        all_session_metrics.append(session_mean_reward)
        print(f'Step {step_idx} sess {session_in_step} (abs {session_abs}): '
              f'reward={session_mean_reward:.4f}, grad_norm={float(np.mean(update_grad_norms)):.4f}')

        sn = zero_pad(session_in_step, 6)
        with open(step_rewards_dir / sn, 'ab') as f:
            np.save(f, np.array(update_rewards))

        if should_save(session_in_step, step.n_sessions, step.output_save_start,
                       step.output_save_step, step.output_save_end):
            _save_rl_trajectory(config, train_state, sp_reset_fn, env_params, step_traj_dir,
                                session_abs, single_patch=True, patch_type=patch_type)
            checkpoints.save_checkpoint(
                ckpt_dir=str(step_weights_dir),
                target={'params': train_state.params},
                step=session_abs, overwrite=False, keep=float('inf'),
            )
            print(f'  -> Saved weights to {step_weights_dir}/checkpoint_{session_abs}')

        session_abs += 1

    return train_state, session_abs


# ----------------------------------------------------------------------------
# Foraging task (full env A2C, reuses the original session loop)
# ----------------------------------------------------------------------------
def run_foraging_step(config, step, step_idx, train_state, env_params, reset_fn,
                      session_abs, all_session_metrics, step_traj_dir,
                      step_weights_dir, step_rewards_dir):
    for session_in_step in trange(step.n_sessions, desc=f'Step {step_idx} (foraging)'):
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

        train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            critic_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            prev_action=jnp.zeros((config.num_envs,), dtype=jnp.int32),
            prev_reward=jnp.zeros((config.num_envs,)),
            prev_obs=obs,
        )

        train_state, env_states, all_metrics = run_session_updates_with_metrics(
            train_state=train_state,
            env_states=env_states,
            env_params=env_params,
            gamma=config.gamma,
            critic_weight=config.critic_weight,
            entropy_weight=config.entropy_weight,
            env_prediction_weight=config.env_prediction_weight,
            global_reward_weight=step.global_reward_weight,
            activity_norm_weight=config.activity_norm_weight,
            pred_obs_weight=config.pred_obs_weight,
            input_noise_std=config.input_noise_std,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            unit_noise_std=config.unit_noise_std,
            rnn_type=config.rnn_type,
            obs_size=config.obs_size,
        )

        avg_rewards = all_metrics['mean_reward']
        grad_norms = all_metrics['grad_norm']
        session_mean_reward = float(jnp.mean(avg_rewards))
        all_session_metrics.append(session_mean_reward)
        print(f'Step {step_idx} sess {session_in_step} (abs {session_abs}): '
              f'reward={session_mean_reward:.4f}, grad_norm={float(jnp.mean(grad_norms)):.4f}')

        sn = zero_pad(session_in_step, 6)
        with open(step_rewards_dir / sn, 'ab') as f:
            np.save(f, np.array(avg_rewards))

        if should_save(session_in_step, step.n_sessions, step.output_save_start,
                       step.output_save_step, step.output_save_end):
            _save_rl_trajectory(config, train_state, reset_fn, env_params, step_traj_dir,
                                session_abs, single_patch=False)
            checkpoints.save_checkpoint(
                ckpt_dir=str(step_weights_dir),
                target={'params': train_state.params},
                step=session_abs, overwrite=False, keep=float('inf'),
            )
            print(f'  -> Saved weights to {step_weights_dir}/checkpoint_{session_abs}')

        session_abs += 1

    return train_state, session_abs


def _save_rl_trajectory(config, train_state, reset_fn, env_params, step_traj_dir,
                        session_abs, single_patch, patch_type=0):
    rng_key, save_reset_key = random.split(train_state.rng_key)
    train_state = train_state.replace(rng_key=rng_key)
    save_obs, save_env_states = jax.vmap(reset_fn, in_axes=(0, None))(
        save_reset_key[None], env_params
    )
    save_train_state = train_state.replace(
        actor_hidden=jnp.zeros((1, config.hidden_size)),
        critic_hidden=jnp.zeros((1, config.hidden_size)),
        prev_action=jnp.zeros((1,), dtype=jnp.int32),
        prev_reward=jnp.zeros((1,)),
        prev_obs=save_obs,
    )
    n_steps = N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE
    if single_patch:
        trajectory, _, _ = collect_trajectory_single_patch(
            train_state=save_train_state, env_states=save_env_states, env_params=env_params,
            input_noise_std=config.input_noise_std, unit_noise_std=config.unit_noise_std,
            rnn_type=config.rnn_type, hidden_size=config.hidden_size,
            obs_size=config.obs_size, n_steps=n_steps, patch_type=patch_type,
        )
    else:
        trajectory, _, _ = collect_trajectory(
            train_state=save_train_state, env_states=save_env_states, env_params=env_params,
            input_noise_std=config.input_noise_std, unit_noise_std=config.unit_noise_std,
            rnn_type=config.rnn_type, hidden_size=config.hidden_size,
            obs_size=config.obs_size, n_steps=n_steps,
        )
    traj_no_batch = jax.tree_util.tree_map(lambda x: x[0], trajectory)
    traj_dict = serialization.to_state_dict(traj_no_batch)
    traj_path = step_traj_dir / f'traj_{session_abs:06d}.pkl'
    with open(traj_path, 'wb') as f:
        pickle.dump([traj_dict], f)
    print(f'  -> Saved trajectory to {traj_path}')


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def train_curriculum(config: CurriculumConfig):
    total_sessions = sum(s.n_sessions for s in config.curriculum)
    tasks = [s.task for s in config.curriculum]
    print(f"Starting pretraining curriculum: {config.exp_name}")
    print(f"  {len(config.curriculum)} steps ({' -> '.join(tasks)}), {total_sessions} total sessions")
    print(f"  Num envs: {config.num_envs}, hidden: {config.hidden_size}")

    rng = np.random.default_rng(config.seed)
    rng_key = random.key(config.seed)
    net_init_key, head_key, rng_key = random.split(rng_key, 3)

    network, params = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        unit_noise_std=config.unit_noise_std,
        rng_key=net_init_key,
        init_scale=config.init_scale,
    )
    # Add the integration readout head so it exists in the param tree throughout.
    params = add_integration_head(network, params, head_key, config.hidden_size)

    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        hidden_size=config.hidden_size,
        num_envs=config.num_envs,
        learning_rate=config.curriculum[0].learning_rate,
        params=params,
    )

    save_dir = Path(f"checkpoints/{config.exp_name}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    rewards_dir = save_dir / '_reward_rates'
    rewards_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path('results') / config.exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    reset_fn, _, _ = TreadmillEnvironment()

    all_session_metrics = []
    task_segments = []  # (task, global_start, global_end) per curriculum step
    session_abs = 0

    for step_idx, step in enumerate(config.curriculum):
        print(f"\n{'='*60}")
        print(f"Curriculum step {step_idx}: task={step.task}, {step.n_sessions} sessions")

        train_state = train_state.replace(learning_rate=step.learning_rate)

        # Reinitialize input weights + actor logits head at each transition.
        if step_idx > 0 and config.reinit_io_at_transition:
            rng_key, reinit_key = random.split(train_state.rng_key)
            new_params = reinitialize_io_weights(config, train_state.params, reinit_key)
            train_state = train_state.replace(
                params=new_params,
                opt_state=init_opt(new_params, step.learning_rate),
                rng_key=rng_key,
            )
            print("  -> Reinitialized input weights + actor logits head (and optimizer state)")

        step_traj_dir = results_dir / f'step_{step_idx:02d}'
        step_traj_dir.mkdir(parents=True, exist_ok=True)
        step_weights_dir = save_dir / f'step_{step_idx:02d}'
        step_weights_dir.mkdir(parents=True, exist_ok=True)
        step_rewards_dir = rewards_dir / f'step_{step_idx:02d}'
        step_rewards_dir.mkdir(parents=True, exist_ok=True)

        seg_start = len(all_session_metrics)

        if step.task == 'integration':
            train_state, session_abs = run_integration_step(
                config, step, step_idx, train_state, rng, session_abs,
                all_session_metrics, step_weights_dir, step_rewards_dir,
            )
        elif step.task == 'single_patch':
            env_params = build_single_patch_env_params(step)
            train_state, session_abs = run_single_patch_step(
                config, step, step_idx, train_state, env_params, reset_fn,
                session_abs, all_session_metrics, step_traj_dir,
                step_weights_dir, step_rewards_dir,
            )
        elif step.task == 'foraging':
            env_params = build_env_params(step)
            train_state, session_abs = run_foraging_step(
                config, step, step_idx, train_state, env_params, reset_fn,
                session_abs, all_session_metrics, step_traj_dir,
                step_weights_dir, step_rewards_dir,
            )
        else:
            raise ValueError(f"Unknown task '{step.task}' in curriculum step {step_idx}")

        task_segments.append((step.task, seg_start, len(all_session_metrics)))

        # --- plot: integration MSE on top, reward rate on bottom ---
        has_integ = any(t == 'integration' for t, _, _ in task_segments)
        has_reward = any(t != 'integration' for t, _, _ in task_segments)
        n_panels = (1 if has_integ else 0) + (1 if has_reward else 0)

        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), squeeze=False)
        ax_integ = axes[0, 0] if has_integ else None
        ax_rew = axes[-1, 0] if has_reward else None

        boundaries = [sum(s.n_sessions for s in config.curriculum[:k])
                      for k in range(1, step_idx + 1)]

        for task, s0, s1 in task_segments:
            xs = list(range(s0, s1))
            ys = all_session_metrics[s0:s1]
            if task == 'integration':
                # stored as negative MSE; plot as positive MSE
                ax_integ.plot(xs, [-v for v in ys], color='purple', linewidth=0.8)
            else:
                color = 'steelblue' if task == 'single_patch' else 'darkorange'
                ax_rew.plot(xs, ys, color=color, linewidth=0.8, label=task)

        for bnd in boundaries:
            if ax_integ is not None:
                ax_integ.axvline(bnd, color='gray', linestyle='--', alpha=0.5)
            if ax_rew is not None:
                ax_rew.axvline(bnd, color='gray', linestyle='--', alpha=0.5)

        if ax_integ is not None:
            ax_integ.set_ylabel('Integration MSE')
            ax_integ.set_title(f'{config.exp_name} — after step {step_idx} ({step.task})')
        if ax_rew is not None:
            ax_rew.set_ylabel('Reward rate')
            ax_rew.set_xlabel('Session (global)')
            ax_rew.legend(fontsize=8)
            if ax_integ is None:
                ax_rew.set_title(f'{config.exp_name} — after step {step_idx} ({step.task})')

        fig.tight_layout()
        fig.savefig(results_dir / 'reward_rate.png', dpi=100)
        plt.close(fig)

    print("\nCurriculum pretraining complete!")
    return train_state, all_session_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to curriculum JSON config file')
    parser.add_argument('--n_networks', type=int, default=1,
                        help='Number of networks to train sequentially (default: 1)')
    args = parser.parse_args()

    config = load_curriculum_config(args.config)

    for network_idx in range(args.n_networks):
        network_seed = config.seed + network_idx
        network_exp_name = (
            f"{config.exp_name}_net{network_idx}" if args.n_networks > 1
            else config.exp_name
        )
        network_config = dataclasses.replace(
            config, seed=network_seed, exp_name=network_exp_name
        )
        train_curriculum(network_config)


if __name__ == '__main__':
    main()
