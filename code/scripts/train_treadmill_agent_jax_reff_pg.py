from dis import dis
import sys
import os
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

# Fix for JAX/Optax version compatibility
import jax
import jax.numpy as jnp
# jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_debug_nans', True)

# Handle DeviceArray deprecation
if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

# external imports
from jax import random, lax
from flax import struct, serialization
from flax.training import checkpoints
from flax import linen as nn
from flax.traverse_util import flatten_dict
import optax
from typing import Tuple, Dict, Any, Optional, List
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import trange
import argparse
import pickle
import json
from datetime import datetime
from pprint import pprint
from enum import IntEnum

# internal imports
from aux_funcs import zero_pad
from agents.a2c_rnn_reward_pred_flax import A2CRNNFlax, init_network_and_params
from environments.components.train_state import TrainState, create_train_state, init_opt, make_optimizer
from environments.components.treadmill_trajectory import TrajectoryData
# Import your existing JAX environment
from environments.treadmill_env_jax import (
    TreadmillEnvironment, 
    TreadmillEnvParams, 
    TreadmillEnvState, 
    treadmill_session_default_params
)


# Enums for reward configuration
class RewardParamStyle(IntEnum):
    FIXED = 0
    INDEP = 1
    COUPLED = 2
    PER_PATCH_INDEP = 3
    PER_PATCH_INDEP_FIXED_OFFSET = 4


class RewardFuncType(IntEnum):
    EXP = 0
    BLOCK = 1
    MARKOV = 2


# Compile-time constants for JAX JIT compatibility
N_UPDATES_PER_SESSION = 20
N_STEPS_PER_UPDATE = 1000


# --- RPE-based credit assignment (k-window reward-predictor critic) ---
# Critic predicts r_t from the last RPE_K observations (no actions, no rewards). The reward-
# prediction error is exponentially filtered (lambda = e^{-1/RPE_TAU}) into credit, anchored at
# each action and summed over horizon RPE_H -> a fixed forward kernel on the RPE (see the loss).
RPE_K = 4              # context window length (observations only)
RPE_H = 20             # credit horizon
REWARD_PRED_HIDDEN = 64

# The credit feature at each anchor is the flattened stack of the last N_OBS_TIMESTEPS (obs+reward)
# vectors (sizes the per-action credit matrices carried in train_state.action_elig / action_credit_*).
N_OBS_TIMESTEPS = 1
# Actor single-timestep reward predictor E[r_t | F_{t-1}] (readout on the actor hidden state), weight
# on its MSE loss.
REWARD_PRED_WEIGHT = 0.1 # was 0.1
# Value-bootstrap horizon k for the advantage A_t = M_t * sum_{s=t}^{t+K} gamma^(s-t) r_s + gamma^K V_{t+K} - V_t.
# V (critic head) supplies the baseline and the bootstrap; requires critic_weight > 0 to train V.
K_BOOT = 20
# NOTE: the M-modulation leak and overall scale are now CONFIG fields, config.m_decay / config.m_scale
# (threaded into collect_trajectory / compute_a2c_loss). M = 1 - m_scale * (L_non_belief + 0.5 <belief, L_belief>): a
# belief-INDEPENDENT baseline L_non_belief (leaky sum E[r]) plus the belief-weighted S_C gating term L_belief. Both
# decay by m_decay/step and reset at each reward; see the loss for the full definition.
# Belief head: the critic produces a hidden_size belief b_t; a feedforward net predicts the next
# N_BELIEF_PREDICT [obs, reward] tuples from it (belief-prediction MSE, weight BELIEF_PRED_WEIGHT).
# S_C(b_l, b_t) = 1 + 0.5*cos_sim(b_l, b_t) in [0.5, 1.5] gates each past E[r_l] penalty in M: only
# carry a missed-reward penalty forward if the context (belief) still matches -> a learned soft reset
# at context/patch boundaries. Uses the unit-normalized-belief identity to keep M an O(N) leaky scan.
N_BELIEF_PREDICT = 20
BELIEF_DIM = 32
BELIEF_PRED_HIDDEN = 64
BELIEF_PRED_WEIGHT = 1.0


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'n_steps', 'obs_size', 'intervention_fn', 'use_belief'])
def collect_trajectory(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    input_noise_std: float,
    unit_noise_std: float,
    rnn_type: str,
    hidden_size: int,
    obs_size: int,
    n_steps: int,
    m_decay: float = 0.99,
    m_scale: float = 0.5,
    L_non_belief_in=None,
    L_belief_in=None,
    intervention_fn=None,
    use_belief: bool = True,
) -> Tuple[TrajectoryData, TrainState, TreadmillEnvState, jnp.ndarray, jnp.ndarray]:
    """Collect trajectory using lax.scan over time steps.

    L_non_belief_in / L_belief_in seed the M-modulation accumulators (leaky sum E[r] and its
    belief-weighted counterpart) so the trace is CONTINUOUS across chunks; pass None to start from
    zero. Returns the final accumulator values so the caller can thread them into the next chunk."""

    network = A2CRNNFlax(
        action_size=2,  # Fixed ACTION_SIZE
        hidden_size=hidden_size,
        unit_noise_std=unit_noise_std,
        rnn_type=rnn_type,
        obs_size=obs_size,
        belief_dim=BELIEF_DIM,
        belief_pred_hidden=BELIEF_PRED_HIDDEN,
        belief_pred_horizon=N_BELIEF_PREDICT,
    )

    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()

    step_num = jnp.zeros_like(env_states.exp_filtered_reward_rate)  # (num_envs,)
    n_envs = env_states.exp_filtered_reward_rate.shape[0]
    # seed the accumulators from the caller (threaded across chunks) or from zero
    L_non_belief0 = jnp.zeros((n_envs,)) if L_non_belief_in is None else L_non_belief_in   # scalar baseline (leaky sum E[r])
    L_belief0 = jnp.zeros((n_envs, BELIEF_DIM)) if L_belief_in is None else L_belief_in    # belief-weighted vector (S_C)

    def scan_step(carry, _):
        train_state, env_states, step_num, L_non_belief, L_belief = carry
        rng_key = train_state.rng_key

        # Sample actions using current observations (from previous step)
        prev_action_one_hot = jax.nn.one_hot(train_state.prev_action, num_classes=2)
        network_input = jnp.concatenate([
            train_state.prev_obs,
            prev_action_one_hot,
            train_state.prev_reward[..., None],
        ], axis=-1)

        # Add input noise
        rng_key, noise_key = random.split(rng_key)
        obs_noise = random.normal(noise_key, network_input.shape) * input_noise_std
        network_input = network_input + obs_noise

        rng_key, network_noise_key = random.split(rng_key)

        # jax.debug.print('{x}', x=network_input[0, :])

        # Forward pass through network
        logits, values, new_actor_hidden, new_critic_hidden, pred_env_quality, pred_obs, pred_reward_rate, belief_pred, belief, reward_pred_1step = network.apply(
            train_state.params,
            jax.lax.stop_gradient(network_input),
            train_state.actor_hidden,
            train_state.critic_hidden,
            rngs={'noise': network_noise_key} if train_state.params else {}
        )

        # Sample actions
        rng_key, action_key = random.split(rng_key)
        action_keys = random.split(action_key, logits.shape[0])
        actions = jax.vmap(
            lambda key, logit: random.categorical(key, logit)
        )(action_keys, logits)

        # Step environments with sampled actions
        rng_key, step_key = random.split(rng_key)
        step_keys = random.split(step_key, actions.shape[0])
        step_results = jax.vmap(
            lambda key, state, action: step_fn(key, state, action, env_params)
        )(step_keys, env_states, actions)

        new_obs, new_env_states, rewards, dones, infos = step_results

        # Update train state with new info. (Per-(obs, action) credit is computed downstream in
        # compute_a2c_loss, which reuses the reward-predictor net; the filter state carried in
        # train_state.action_elig / action_credit_* is advanced there.)
        new_train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=new_actor_hidden,
            critic_hidden=new_critic_hidden,
            prev_obs=new_obs,
            prev_action=actions,
            prev_reward=rewards,
        )

        # Running modulation M = 1 - m_scale * (L_non_belief + 0.5 <belief_t, L_belief_t>), the S_C-gated leaky sum
        # since the last reward (reset after reward, pre-update carry). L_non_belief is the belief-INDEPENDENT
        # baseline (leaky sum E[r]); the 0.5<belief, L_belief> term adds the context-similarity gating. E[r_t] is
        # the dedicated 1-step head. Matches the loss exactly.
        er0 = reward_pred_1step                                            # E[r_t] from the dedicated 1-step head
        belief_sc = belief if use_belief else jnp.zeros_like(belief)       # ablate: no S_C gating (baseline survives)
        M = 1.0 - m_scale * (L_non_belief + 0.5 * jnp.sum(belief_sc * L_belief, axis=-1))
        keep = jnp.where(rewards > 0, 0.0, m_decay)                        # reset after reward
        L_non_belief_new = er0 + keep * L_non_belief                                     # belief-independent baseline
        L_belief_new = er0[:, None] * belief_sc + keep[:, None] * L_belief               # belief is unit-norm

        # Return step data (logits stored are the gated logits actually used for decisions)
        step_data = {
            'observations': network_input,
            'actions': actions,
            'rewards': rewards,
            'logits': logits,
            'values': values,
            'dones': dones,
            'actor_hidden': train_state.actor_hidden,
            'critic_hidden': train_state.critic_hidden,
            'pred_environment_quality': pred_env_quality,
            'pred_obs': pred_obs,
            'exp_filtered_reward_rate': new_env_states.exp_filtered_reward_rate,
            'pred_reward_rate': pred_reward_rate,
            'belief_pred': belief_pred,   # (num_envs, N_BELIEF_PREDICT*(obs+1)) belief [obs, reward] forecast
            'reward_pred_1step': reward_pred_1step,   # (num_envs,) dedicated E[r_t] head (M's E[r])
            'M': M,                       # (num_envs,) reward modulation M (stored under legacy 'M' field)
            'belief': belief,             # (num_envs, BELIEF_DIM) critic belief vector for S_C context gating
        } | infos

        return (new_train_state, new_env_states, jnp.zeros(rewards.shape[0]), L_non_belief_new, L_belief_new), step_data

    # Run scan over time steps using compile-time constant
    (final_train_state, final_env_states, _, L_non_belief_final, L_belief_final), trajectory_data = lax.scan(
        scan_step,
        (train_state, env_states, step_num, L_non_belief0, L_belief0),
        None,
        length=n_steps
    )

    # Reshape trajectory data from (n_steps, num_envs, ...) to (num_envs, n_steps, ...)
    trajectory_data = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectory_data
    )

    trajectory = TrajectoryData(**trajectory_data)

    return trajectory, final_train_state, final_env_states, L_non_belief_final, L_belief_final


def compute_a2c_loss(
    params: Any,
    train_state: Any,
    env_states: Any,
    env_params: Any,
    gamma: float,
    critic_weight: float,
    entropy_weight: float,
    env_prediction_weight: float,
    global_reward_weight: float,
    activity_norm_weight: float,
    pred_obs_weight: float,
    input_noise_std: float,
    hidden_size: int,
    unit_noise_std: float,
    rnn_type: str,
    obs_size: int,
    m_decay: float = 0.99,
    m_scale: float = 0.5,
    L_non_belief_in=None,
    L_belief_in=None,
    use_belief: bool = True,
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], Any, Any, jnp.ndarray, jnp.ndarray]]:
    """Compute A2C loss; collect_trajectory is called here so BPTT flows through the scan.

    L_non_belief_in / L_belief_in seed the M accumulators (chunk-start values, threaded across chunks);
    they seed BOTH the collect-side scan and the loss-side _cum_scan so M is continuous. The final
    (chunk-end) accumulators are returned in the aux tuple (stop-gradded) for the next chunk."""

    trajectory, final_train_state, final_env_states, L_non_belief_final, L_belief_final = collect_trajectory(
        train_state=train_state.replace(params=params),
        env_states=env_states,
        env_params=env_params,
        input_noise_std=input_noise_std,
        unit_noise_std=unit_noise_std,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        obs_size=obs_size,
        n_steps=N_STEPS_PER_UPDATE,
        m_decay=m_decay,
        m_scale=m_scale,
        L_non_belief_in=L_non_belief_in,
        L_belief_in=L_belief_in,
        use_belief=use_belief,
    )

    logits = trajectory.logits            # (B, T, A)
    rewards = trajectory.rewards          # (B, T)
    N = N_STEPS_PER_UPDATE                 # chunk length (== T)
    B = rewards.shape[0]                    # num envs

    # belief prediction loss: predict the next N_BELIEF_PREDICT [obs, reward] tuples from the critic belief b_t
    belief_pred = trajectory.belief_pred.reshape(B, N, N_BELIEF_PREDICT, obs_size + 1)  # (B, N, NB, obs+1)
    bj = jnp.clip(jnp.arange(N)[:, None] + jnp.arange(N_BELIEF_PREDICT)[None, :], 0, N - 1)  # (N, NB)  t..t+NB-1
    obs_tgt = lax.stop_gradient(trajectory.observations[:, :, :obs_size])[:, bj]        # (B, N, NB, obs)
    rew_tgt = lax.stop_gradient(rewards)[:, bj][..., None]                              # (B, N, NB, 1)
    belief_target = jnp.concatenate([obs_tgt, rew_tgt], axis=-1)                        # (B, N, NB, obs+1)
    belief_pred_loss = jnp.mean((belief_pred - belief_target) ** 2)

    # dedicated 1-step reward head E[r_t]
    reward_pred_1step = trajectory.reward_pred_1step                      # (B, N)  E[r_t]
    reward_pred_1step_loss = jnp.mean((reward_pred_1step - lax.stop_gradient(rewards)) ** 2)

    rp_sg = lax.stop_gradient(reward_pred_1step)                         # (B, N)  E[r_t] for M
    rewards_sg = lax.stop_gradient(rewards)
    is_reward = rewards_sg > 0                                           # (B, N)

    b_hat = lax.stop_gradient(trajectory.belief)                        # (B, N, HID)
    if not use_belief:
        b_hat = jnp.zeros_like(b_hat)                                   # ablate: no S_C gating (baseline survives)
    b_hat = b_hat / (jnp.linalg.norm(b_hat, axis=-1, keepdims=True) + 1e-8)  # unit (stays zero when ablated)
    def _cum_scan(carry, xt):
        L_non_belief, L_belief = carry                     # L_non_belief: (B,)  L_belief: (B, HID)  pre-update
        rp_k, bhat_k, isr = xt
        contrib = rp_k[:, None] * bhat_k                                # E[r_k] * b_hat_k
        keep = jnp.where(isr, 0.0, m_decay)                            # (B,)  reset after reward
        return (rp_k + keep * L_non_belief, contrib + keep[:, None] * L_belief), (L_non_belief, L_belief)
    # seed from the threaded chunk-start accumulators (same seeds the collect-side scan used), so the
    # advantage's M is continuous across chunks. stop-gradded -> no BPTT across the chunk boundary.
    L_non_belief_seed = jnp.zeros((B,)) if L_non_belief_in is None else lax.stop_gradient(L_non_belief_in)
    L_belief_seed = jnp.zeros((B, BELIEF_DIM)) if L_belief_in is None else lax.stop_gradient(L_belief_in)
    _, (L_non_belief, L_belief) = lax.scan(
        _cum_scan, (L_non_belief_seed, L_belief_seed),
        (jnp.swapaxes(rp_sg, 0, 1),
         jnp.swapaxes(b_hat, 0, 1), jnp.swapaxes(is_reward, 0, 1)))
    L_non_belief = jnp.swapaxes(L_non_belief, 0, 1)   # (B, N)  scalar baseline accumulator (belief-independent)
    L_belief     = jnp.swapaxes(L_belief, 0, 1)       # (B, N, HID)  belief-weighted vector accumulator
    M = 1.0 - m_scale * (L_non_belief + 0.5 * jnp.sum(b_hat * L_belief, axis=-1))  # (B, N)  baseline + S_C-gated modulation

    # k-step value-bootstrap advantage (see note #1). G_t = sum_{i=0}^{K} gamma^i r_{t+i}; near the
    # chunk end the window and bootstrap shrink to the actual remaining horizon `gap` so they stay
    # consistent. M_t gates G_t; V provides the -V_t baseline and the gamma^gap V_{t+gap} bootstrap.
    V = trajectory.values                                              # (B, N)  V_t (critic head)
    gap    = jnp.minimum(K_BOOT, N - 1 - jnp.arange(N)).astype(jnp.float32)   # (N,)  per-t bootstrap horizon
    lags   = jnp.arange(K_BOOT + 1)                                    # i = 0..K
    fut    = jnp.arange(N)[:, None] + lags[None, :]                    # (N, K+1)  t+i
    idx    = jnp.clip(fut, 0, N - 1)                                   # (N, K+1)
    inb    = (fut < N).astype(jnp.float32)[None]                       # (1, N, K+1)  in-chunk mask
    gpow   = (gamma ** lags)[None, None, :]                            # (1,1,K+1)
    G      = jnp.sum(rewards_sg[:, idx] * gpow * inb, axis=-1)         # (B, N)  sum_i gamma^i r_{t+i}
    boot_i = (jnp.arange(N) + gap).astype(jnp.int32)                  # = min(t+K, N-1)
    v_boot = lax.stop_gradient(V)[:, boot_i]                          # (B, N)  V_{t+gap}
    td_target = lax.stop_gradient(M) * G + (gamma ** gap)[None, :] * v_boot   # (B, N)  M_t G_t + gamma^K V_{t+K}
    critic_loss = jnp.mean((V - lax.stop_gradient(td_target)) ** 2)    # fit V to the M-modulated TD target
    coeff = lax.stop_gradient(td_target - V)                          # (B, N)  A_t, actor weight on log pi

    log_probs = jax.nn.log_softmax(logits)
    chosen_log_probs = jnp.take_along_axis(
        log_probs,
        lax.stop_gradient(trajectory.actions[..., None]),
        axis=-1,
    ).squeeze(-1)

    actor_loss = -jnp.mean(coeff * chosen_log_probs)                     # averaged over the chunk (B x N)

    # jax.debug.print('{x}', x=critic_loss)

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
        + BELIEF_PRED_WEIGHT * belief_pred_loss
        + REWARD_PRED_WEIGHT * reward_pred_1step_loss
        + entropy_weight * entropy_loss
        + activity_norm_weight * activity_norm
    )

    metrics = {
        'total_loss': total_loss,
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'belief_pred_loss': belief_pred_loss,
        'reward_pred_1step_loss': reward_pred_1step_loss,
        'entropy_loss': entropy_loss,
        'activity_loss': activity_norm,
        'mean_reward': jnp.mean(trajectory.rewards),
        'advantage_mean': jnp.mean(coeff),
        'advantage_std': jnp.std(coeff),
        'value_mean': jnp.mean(V),
        'M_mean': jnp.mean(M),
        'M_frac_neg': jnp.mean((M < 0).astype(jnp.float32)),   # fraction of steps that are "overdue"
    }

    return total_loss, (metrics, jax.lax.stop_gradient(final_train_state), jax.lax.stop_gradient(final_env_states),
                        jax.lax.stop_gradient(L_non_belief_final), jax.lax.stop_gradient(L_belief_final))


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'obs_size', 'use_belief'])
def train_step(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    critic_weight: float,
    entropy_weight: float,
    env_prediction_weight: float,
    global_reward_weight: float,
    activity_norm_weight: float,
    pred_obs_weight: float,
    input_noise_std: float,
    action_size: int,
    hidden_size: int,
    unit_noise_std: float,
    rnn_type: str,
    obs_size: int,
    m_decay: float = 0.99,
    m_scale: float = 0.5,
    L_non_belief_in=None,
    L_belief_in=None,
    use_belief: bool = True,
) -> Tuple[TrainState, TreadmillEnvState, Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Single training step. L_non_belief_in / L_belief_in seed the M accumulators (threaded across
    chunks); the final accumulators are returned for the next chunk."""

    grad_fn = jax.grad(compute_a2c_loss, has_aux=True)
    grads, (metrics, final_train_state, final_env_states, L_non_belief_final, L_belief_final) = grad_fn(
        train_state.params,
        train_state,
        env_states,
        env_params,
        gamma,
        critic_weight,
        entropy_weight,
        env_prediction_weight,
        global_reward_weight,
        activity_norm_weight,
        pred_obs_weight,
        input_noise_std,
        hidden_size,
        unit_noise_std,
        rnn_type,
        obs_size,
        m_decay,
        m_scale,
        L_non_belief_in,
        L_belief_in,
        use_belief,
    )

    metrics['grad_norm'] = optax.global_norm(grads)

    # Apply updates (critic/reward-predictor params use a scaled learning rate; must match init_opt)
    optimizer = make_optimizer(
        train_state.params, train_state.learning_rate, train_state.reward_pred_lr_scale
    )
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)

    # Update training state
    final_train_state = final_train_state.replace(
        params=new_params,
        opt_state=new_opt_state,
    )
    
    return final_train_state, final_env_states, metrics, L_non_belief_final, L_belief_final

# Configuration matching your original hyperparameters
@struct.dataclass
class TrainingConfig:


    # Environment
    exp_name: str = ''
    num_envs: int = 64
    patch_types_per_env: int = 3
    obs_size: int = 4  # patch_types_per_env + 1
    action_size: int = 2
    dwell_time_for_reward: int = 3
    reward_site_len: int = 3
    input_noise_std: float = 1e-2
    unit_noise_std: float = 1e-2
    reward_param_style: str = 'fixed'
    reward_func_type: str = 'exp'
    reward_decay_consts: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0.0, 10.0, 30.0]))
    reward_prob_prefactors: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0.8, 0.8, 0.8]))
    fixed_patches: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0, 0, 0]))

    reward_decay_range: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0.0, 40.0]))
    reward_prob_range: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0.0, 1.0]))
    patch_active_transition_prob_range: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([0.9, 0.9]))
    interreward_len_bounds: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([1.0, 6.0]))
    interreward_len_decay_rate: float = 0.8
    interpatch_len_bounds: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([1.0, 12.0]))
    interpatch_len_decay_rate: float = 0.1

    # Agent params
    hidden_size: int = 128
    critic_weight: float = 0.0785
    entropy_weight: float = 1.02e-6 # 1.02e-06
    env_prediction_weight: float = 0 # 0.001
    global_reward_weight: float = 0
    activity_norm_weight: float = 1e-4
    pred_obs_weight: float = 0
    gamma: float = 0.999 # 0.987
    learning_rate: float = 2.5e-5 # 1e-4
    reward_pred_lr_scale: float = 0.1  # reward-predictor LR = learning_rate * this; <1 -> slower
    m_decay: float = 0.99  # M-modulation leak: per-step decay of L_non_belief/L_belief accumulators (reset at rewards)
    m_scale: float = 0.5   # overall scale on the M modulation: M = 1 - m_scale * (L_non_belief + 0.5 <belief, L_belief>)
    rnn_type: str = 'GRU'
    init_scale: float = 1.0

    # Training params (runtime configurable)
    seed: int = 0
    n_sessions: int = 5000

    # Belief gating
    use_belief: bool = True

    # Logging
    output_state_save_rate: int = 100

    # Periodic in-training trajectory/weight snapshots (see should_save)
    n_save_envs: int = 1
    output_save_start: int = -1
    output_save_step: int = 1
    output_save_end: Optional[int] = None


def save_config_to_json(config: TrainingConfig, filepath: str) -> None:
    """Save TrainingConfig to JSON."""
    config_dict = serialization.to_state_dict(config)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def should_save(session_num: int, n_sessions: int, save_start: int,
                save_step: int, save_end: Optional[int]) -> bool:
    """Whether to save a trajectory/weight snapshot at this session.

    save_start == -1 saves only the final session; otherwise saves every
    save_step sessions in [save_start, save_end) (save_end defaults to n_sessions).
    """
    if save_start == -1:
        return session_num == n_sessions - 1
    end = save_end if save_end is not None else n_sessions
    return (save_start <= session_num < end and
            (session_num - save_start) % save_step == 0)


def load_config_from_json(filepath: str) -> TrainingConfig:
    """Load TrainingConfig from JSON.
    Missing fields will use their defaults from TrainingConfig."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    # Handle legacy integer values for reward_param_style and reward_func_type
    if 'reward_param_style' in config_dict and isinstance(config_dict['reward_param_style'], int):
        config_dict['reward_param_style'] = RewardParamStyle(config_dict['reward_param_style']).name.lower()
    if 'reward_func_type' in config_dict and isinstance(config_dict['reward_func_type'], int):
        config_dict['reward_func_type'] = RewardFuncType(config_dict['reward_func_type']).name.lower()

    # Convert list fields to JAX arrays
    if 'reward_decay_consts' in config_dict:
        config_dict['reward_decay_consts'] = jnp.array(config_dict['reward_decay_consts'])
    if 'reward_prob_prefactors' in config_dict:
        config_dict['reward_prob_prefactors'] = jnp.array(config_dict['reward_prob_prefactors'])
    if 'reward_decay_range' in config_dict:
        config_dict['reward_decay_range'] = jnp.array(config_dict['reward_decay_range'])
    if 'patch_active_transition_prob_range' in config_dict:
        config_dict['patch_active_transition_prob_range'] = jnp.array(config_dict['patch_active_transition_prob_range'])
    if 'interreward_len_bounds' in config_dict:
        config_dict['interreward_len_bounds'] = jnp.array(config_dict['interreward_len_bounds'])
    if 'interpatch_len_bounds' in config_dict:
        config_dict['interpatch_len_bounds'] = jnp.array(config_dict['interpatch_len_bounds'])
    if 'fixed_patches' in config_dict:
        config_dict['fixed_patches'] = jnp.array(config_dict['fixed_patches'])

    # Start with defaults and update with loaded values
    config = TrainingConfig()
    return config.replace(**config_dict)


@partial(jax.jit, static_argnames=['action_size', 'hidden_size', 'unit_noise_std', 'rnn_type', 'obs_size', 'use_belief'])
def run_session_updates_with_metrics(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    critic_weight: float,
    entropy_weight: float,
    env_prediction_weight: float,
    global_reward_weight: float,
    activity_norm_weight: float,
    pred_obs_weight: float,
    input_noise_std: float,
    action_size: int,
    hidden_size: int,
    unit_noise_std: float,
    rnn_type: str,
    obs_size: int,
    m_decay: float = 0.99,
    m_scale: float = 0.5,
    use_belief: bool = True,
) -> Tuple[TrainState, TreadmillEnvState, Dict[str, jnp.ndarray]]:
    """Run all training updates with full metrics collection"""

    # M-modulation accumulators, threaded across the chunks (updates) of THIS session; start from zero
    # each session (aligned with the fresh env reset at session start).
    n_envs = env_states.exp_filtered_reward_rate.shape[0]
    L_non_belief0 = jnp.zeros((n_envs,))
    L_belief0 = jnp.zeros((n_envs, BELIEF_DIM))

    def update_step(carry, _):
        train_state, env_states, L_non_belief, L_belief = carry

        new_train_state, new_env_states, metrics, L_non_belief, L_belief = train_step(
            train_state=train_state,
            env_states=env_states,
            env_params=env_params,
            gamma=gamma,
            critic_weight=critic_weight,
            entropy_weight=entropy_weight,
            env_prediction_weight=env_prediction_weight,
            global_reward_weight=global_reward_weight,
            activity_norm_weight=activity_norm_weight,
            pred_obs_weight=pred_obs_weight,
            input_noise_std=input_noise_std,
            action_size=action_size,
            hidden_size=hidden_size,
            unit_noise_std=unit_noise_std,
            rnn_type=rnn_type,
            obs_size=obs_size,
            m_decay=m_decay,
            m_scale=m_scale,
            L_non_belief_in=L_non_belief,
            L_belief_in=L_belief,
            use_belief=use_belief,
        )

        return (new_train_state, new_env_states, L_non_belief, L_belief), metrics

    # Run scan over all updates (M accumulators thread through the carry)
    (final_train_state, final_env_states, _, _), all_metrics = lax.scan(
        update_step,
        (train_state, env_states, L_non_belief0, L_belief0),
        None,
        length=N_UPDATES_PER_SESSION,
    )

    # jax.debug.print('grad_norm: {x}', x=all_metrics['grad_norm'])
    # jax.debug.print('activity_norm: {x}', x=all_metrics['activity_loss'])

    return final_train_state, final_env_states, all_metrics


def train_a2c_jax(config: TrainingConfig = None, load_path: str = None):
    """Main training function that matches your existing structure"""

    if config is None:
        config = TrainingConfig()

    print("Starting JAX A2C Training...")
    print(f"Num envs: {config.num_envs}")
    print(f"Sessions: {config.n_sessions}")
    print(f"Updates per session: {N_UPDATES_PER_SESSION}")
    print(f"Steps per update: {N_STEPS_PER_UPDATE}")

    # Initialize everything
    rng_key = random.key(config.seed)
    env_params = treadmill_session_default_params()
    env_params = env_params.replace(
        reward_param_style=reward_param_style_str_to_int(config.reward_param_style),
        reward_func_type=reward_func_type_str_to_int(config.reward_func_type),
        fixed_patches=config.fixed_patches,
        reward_decay_consts=config.reward_decay_consts,
        reward_prob_prefactors=config.reward_prob_prefactors,
        reward_decay_range=config.reward_decay_range,
        reward_prob_range=config.reward_prob_range,
        patch_active_transition_prob_range=config.patch_active_transition_prob_range,
        interreward_len_bounds=config.interreward_len_bounds,
        interreward_len_decay_rate=config.interreward_len_decay_rate,
        interpatch_len_bounds=config.interpatch_len_bounds,
        interpatch_len_decay_rate=config.interpatch_len_decay_rate,
        dwell_time_for_reward=config.dwell_time_for_reward,
    )

    net_init_key, rng_key = random.split(rng_key)

    network, params = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        unit_noise_std=config.unit_noise_std,
        rng_key=net_init_key,
        init_scale=config.init_scale,
        belief_dim=BELIEF_DIM,
        belief_pred_hidden=BELIEF_PRED_HIDDEN,
        belief_pred_horizon=N_BELIEF_PREDICT,
    )

    # Create training state
    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        hidden_size=config.hidden_size,
        num_envs=config.num_envs,
        learning_rate=config.learning_rate,
        params=params,
        reward_pred_lr_scale=config.reward_pred_lr_scale,
    )

    # Load pretrained model if path is given
    if load_path is not None:
        print(f"Loading pretrained model from {load_path}")
        restored = checkpoints.restore_checkpoint(ckpt_dir=load_path, target=None)
        # restored can be just params or a dict depending on how saved
        if "params" in restored:
            params = restored["params"]
        else:
            params = restored
        train_state = train_state.replace(params=params)
    
    print(f"Initialized network with {config.hidden_size} hidden units")
    
    # Initialize environments
    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()
    rng_key, reset_key = random.split(train_state.rng_key)
    reset_keys = random.split(reset_key, config.num_envs)
    
    obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)
    train_state = train_state.replace(prev_obs=obs)
    print(f"Initialized {config.num_envs} environments")
    
    # Storage for logging (matching your original structure)
    all_session_rewards = []

    save_dir_rewards = Path(f'exp_reward_rates/{config.exp_name}').resolve()  # makes it absolute
    save_dir_rewards.mkdir(parents=True, exist_ok=True)

    save_dir = Path(f"checkpoints/{config.exp_name}").resolve()  # makes it absolute
    save_dir.mkdir(parents=True, exist_ok=True)

    save_dir_rewards = save_dir / '_reward_rates'
    save_dir_rewards.mkdir(parents=True, exist_ok=True)

    results_dir = Path('results') / config.exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    traj_dir = results_dir / 'trajectories'
    traj_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = save_dir / 'snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    best_session_reward = -np.inf

    # Training loop (outer loop stays in Python for logging)
    for session_num in trange(config.n_sessions, desc='Sessions'):
        
        avg_rewards_per_update = np.empty((config.num_envs, N_UPDATES_PER_SESSION))
        all_info = []

        # Reset environment for new episode
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

        # Reset hidden states and per-action credit filters at the start of each session.
        train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            critic_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            prev_action=jnp.zeros((config.num_envs,), dtype=jnp.int32),
            prev_reward=jnp.zeros((config.num_envs,)),
            prev_obs=obs,
            action_elig=jnp.zeros((config.num_envs, N_OBS_TIMESTEPS * (config.obs_size + 1), config.action_size)),
            action_credit_reward=jnp.zeros((config.num_envs, N_OBS_TIMESTEPS * (config.obs_size + 1), config.action_size)),
            action_credit_pred=jnp.zeros((config.num_envs, N_OBS_TIMESTEPS * (config.obs_size + 1), config.action_size)),
            luck_filter=jnp.zeros((config.num_envs,)),   # reset recent-luck EMA each session
        )

        train_state, env_states, all_metrics = run_session_updates_with_metrics(
            train_state=train_state,
            env_states=env_states,
            env_params=env_params,
            gamma=config.gamma,
            critic_weight=config.critic_weight,
            entropy_weight=config.entropy_weight,
            env_prediction_weight=config.env_prediction_weight,
            global_reward_weight=config.global_reward_weight,
            activity_norm_weight=config.activity_norm_weight,
            pred_obs_weight=config.pred_obs_weight,
            input_noise_std=config.input_noise_std,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            unit_noise_std=config.unit_noise_std,
            rnn_type=config.rnn_type,
            obs_size=config.obs_size,
            m_decay=config.m_decay,
            m_scale=config.m_scale,
            use_belief=config.use_belief,
        )

        # pprint(train_state.params)

        avg_rewards_per_update = all_metrics['mean_reward']
        grad_norms = all_metrics['grad_norm']
        print('grad_norms')
        print('mean:', jnp.mean(grad_norms), 'std:', jnp.std(grad_norms))
        # print(grad_norms)
        # weight_norms = all_metrics['weight_norm']
        # print('weight mean:', jnp.mean(weight_norms), 'std:', jnp.std(weight_norms))
        # losses = all_metrics['total_loss']
        # print('loss mean:', jnp.mean(losses), 'std:', jnp.std(losses))
        # print()

            
        # Session-level logging
        session_mean_reward = np.mean(avg_rewards_per_update)
        all_session_rewards.append(session_mean_reward)
        
        print(f'Session {session_num}: Avg reward = {session_mean_reward:.4f}')
        
        if session_mean_reward > best_session_reward:
            best_session_reward = session_mean_reward
            print(f"New best reward {best_session_reward:.4f} at session {session_num} — saving checkpoint")
            checkpoints.save_checkpoint(
                ckpt_dir=str(save_dir),
                target={"params": train_state.params},
                step=0,
                overwrite=True,
                keep=1,
            )

        if (session_num + 1) % 50 == 0 or session_num == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(all_session_rewards)
            ax.set_xlabel('Session')
            ax.set_ylabel('Mean reward rate')
            ax.set_title(config.exp_name)
            fig.tight_layout()
            fig.savefig(results_dir / 'reward_rate.png', dpi=100)
            plt.close(fig)

        sn = zero_pad(session_num, 6)
        with open(save_dir_rewards / f'{sn}', 'ab') as f:
            np.save(f, avg_rewards_per_update)

        # Periodic snapshot: roll out n_save_envs envs and save their trajectories + weights
        if should_save(session_num, config.n_sessions, config.output_save_start,
                       config.output_save_step, config.output_save_end):
            rng_key, save_reset_key = random.split(train_state.rng_key)
            train_state = train_state.replace(rng_key=rng_key)
            save_reset_keys = random.split(save_reset_key, config.n_save_envs)
            save_obs, save_env_states = jax.vmap(reset_fn, in_axes=(0, None))(
                save_reset_keys, env_params
            )
            save_train_state = train_state.replace(
                actor_hidden=jnp.zeros((config.n_save_envs, config.hidden_size)),
                critic_hidden=jnp.zeros((config.n_save_envs, config.hidden_size)),
                prev_action=jnp.zeros((config.n_save_envs,), dtype=jnp.int32),
                prev_reward=jnp.zeros((config.n_save_envs,)),
                prev_obs=save_obs,
            )
            trajectory, _, _, _, _ = collect_trajectory(
                train_state=save_train_state,
                env_states=save_env_states,
                env_params=env_params,
                input_noise_std=config.input_noise_std,
                unit_noise_std=config.unit_noise_std,
                rnn_type=config.rnn_type,
                hidden_size=config.hidden_size,
                obs_size=config.obs_size,
                n_steps=N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE,
                m_decay=config.m_decay,
                m_scale=config.m_scale,
                use_belief=config.use_belief,
            )
            traj_dicts = [
                serialization.to_state_dict(
                    jax.tree_util.tree_map(lambda x: x[i], trajectory)
                )
                for i in range(config.n_save_envs)
            ]
            traj_path = traj_dir / f'traj_{session_num:06d}.pkl'
            with open(traj_path, 'wb') as f:
                pickle.dump(traj_dicts, f)
            print(f'  -> Saved trajectory ({config.n_save_envs} envs) to {traj_path}')

            checkpoints.save_checkpoint(
                ckpt_dir=str(snapshot_dir),
                target={'params': train_state.params},
                step=session_num,
                overwrite=False,
                keep=float('inf'),
            )
            print(f'  -> Saved weights to {snapshot_dir}/checkpoint_{session_num}')

    print("Training completed!")
    return train_state, all_session_rewards


def evaluate_a2c_jax(config: TrainingConfig, checkpoint_path: str, save_trajectories: bool = False,
                     intervention_points_path: str = None):
    """Evaluate trained A2C agent without gradient updates"""
    
    print("Starting JAX A2C Evaluation...")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Num episodes: {config.n_sessions}")
    print(f"Save trajectories: {save_trajectories}")
    
    # Initialize everything
    rng_key = random.key(config.seed)
    env_params = treadmill_session_default_params()
    env_params = env_params.replace(
        reward_param_style=reward_param_style_str_to_int(config.reward_param_style),
        reward_func_type=reward_func_type_str_to_int(config.reward_func_type),
        fixed_patches=config.fixed_patches,
        reward_decay_consts=config.reward_decay_consts,
        reward_prob_prefactors=config.reward_prob_prefactors,
        reward_decay_range=config.reward_decay_range,
        patch_active_transition_prob_range=config.patch_active_transition_prob_range,
        interreward_len_bounds=config.interreward_len_bounds,
        interreward_len_decay_rate=config.interreward_len_decay_rate,
        interpatch_len_bounds=config.interpatch_len_bounds,
        interpatch_len_decay_rate=config.interpatch_len_decay_rate,
        reward_site_len=config.reward_site_len,
        dwell_time_for_reward=config.dwell_time_for_reward,
    )

    session_steps = N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE

    net_init_key, rng_key = random.split(rng_key)

    network, params = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        unit_noise_std=config.unit_noise_std,
        rng_key=net_init_key,
        init_scale=config.init_scale,
        belief_dim=BELIEF_DIM,
        belief_pred_hidden=BELIEF_PRED_HIDDEN,
        belief_pred_horizon=N_BELIEF_PREDICT,
    )

    # Create training state (just for structure, won't be updated)
    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        hidden_size=config.hidden_size,
        num_envs=1,  # Use single environment for cleaner episode tracking
        learning_rate=config.learning_rate,
        params=params,
        reward_pred_lr_scale=config.reward_pred_lr_scale,
    )

    # Load trained model
    print(f"Loading trained model from {checkpoint_path}")
    restored = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
    if "params" in restored:
        params = restored["params"]
    else:
        params = restored
    train_state = train_state.replace(params=params)
    print("Model loaded successfully")
    
    # Initialize environment
    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()
    
    # Build intervention closure if a points file was provided
    intervention_fn = None
    if intervention_points_path is not None:
        with open(intervention_points_path, 'rb') as f:
            _points = jnp.array(pickle.load(f))   # (N, H)
        print(f"Loaded intervention points: shape {_points.shape} from {intervention_points_path}")
        def intervention_fn(actor_hidden, _pts=_points, r=0.01):
            diffs = actor_hidden[:, None, :] - _pts[None, :, :]   # (num_envs, N, H)
            dists = jnp.sum(diffs ** 2, axis=-1)                   # (num_envs, N)
            nearest = jnp.argmin(dists, axis=-1)                   # (num_envs,)
            nearest_dist = jnp.sqrt(jnp.min(dists, axis=-1))
            # jax.debug.print('Nearest fixed point dist {x}', x=nearest_sq_dist)
            new_hidden = jnp.where(
                nearest_dist > 1e-8,
                jnp.where(
                    nearest_dist < r,
                    actor_hidden,
                    _pts[nearest] + r * (actor_hidden - _pts[nearest]) / nearest_dist,
                ),
                _pts[nearest],
            )
            # pert = jnp.where(nearest_dist < r, 0, pert)
            return new_hidden # (num_envs, H)

    # Storage for results
    all_episode_rewards = []
    all_trajectories = [] if save_trajectories else None

    # Run evaluation episodes
    for episode in trange(config.n_sessions, desc='Sessions'):

        # Reset environment for new episode
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

        # Reset hidden states
        train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=jnp.zeros((1, config.hidden_size)),
            critic_hidden=jnp.zeros((1, config.hidden_size)),
            prev_action=jnp.zeros((1,), dtype=jnp.int32),
            prev_reward=jnp.zeros((1,)),
            prev_obs=obs,
        )

        # Run episode (using a reasonable episode length)
        trajectory, final_train_state, final_env_states, _, _ = collect_trajectory(
            train_state=train_state,
            env_states=env_states,
            env_params=env_params,
            input_noise_std=0,  # No noise during evaluation
            unit_noise_std=0,
            rnn_type=config.rnn_type,
            hidden_size=config.hidden_size,
            obs_size=config.obs_size,
            n_steps=session_steps,
            m_decay=config.m_decay,
            m_scale=config.m_scale,
            intervention_fn=intervention_fn,
            use_belief=config.use_belief,
        )
        
        # Extract episode metrics
        episode_reward = float(jnp.sum(trajectory.rewards))
        
        all_episode_rewards.append(episode_reward)
        
        # Save trajectory if requested
        if save_trajectories:
            # Convert JAX arrays to numpy for easier saving
            traj_no_batch = jax.tree_util.tree_map(lambda x: x[0], trajectory)
            trajectory_dict = serialization.to_state_dict(traj_no_batch)
            all_trajectories.append(trajectory_dict)
        
        # Update rng for next episode
        train_state = final_train_state
    
    # Compute summary statistics
    mean_reward_rate = np.mean(all_episode_rewards) / session_steps
    std_reward_rate = np.std(all_episode_rewards) / session_steps
    
    print("\nEvaluation Summary:")
    print(f"Mean episode reward rate: {mean_reward_rate:.4f} ± {std_reward_rate:.4f}")
    print(f"Min/Max reward rates: {np.min(all_episode_rewards) / session_steps:.4f} / {np.max(all_episode_rewards) / session_steps:.4f}")
    
    # Save results
    results = {
        'episode_rewards': all_episode_rewards,
        'mean_reward_rate': mean_reward_rate,
        'std_reward_rate': std_reward_rate,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Create results directory
    results_dir = Path(f"results/{config.exp_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary results
    results_file = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_file}")
    
    # Save trajectories if requested
    if save_trajectories and all_trajectories:
        trajectories_file = results_dir / f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(trajectories_file, 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f"Trajectories saved to {trajectories_file}")
    
    return results, all_trajectories


def reward_param_style_str_to_int(style):
    try:
        return RewardParamStyle[style.upper()].value
    except KeyError:
        raise ValueError(f"Unknown reward param style: {style}. Options: {', '.join([e.name.lower() for e in RewardParamStyle])}")


def reward_func_type_str_to_int(func_type):
    try:
        return RewardFuncType[func_type.upper()].value
    except KeyError:
        raise ValueError(f"Unknown reward func type: {func_type}. Options: {', '.join([e.name.lower() for e in RewardFuncType])}")


def train_and_evaluate_network(config: TrainingConfig) -> Tuple[Dict, List]:
    """Train a network and then automatically evaluate it.

    Args:
        config: TrainingConfig for this network

    Returns:
        (results_dict, training_rewards_list)
        - results_dict: results from evaluate_a2c_jax containing eval metrics
        - training_rewards_list: per-session rewards from training
    """
    print(f"\n{'='*60}")
    print(f"Training network with exp_name: {config.exp_name}, seed: {config.seed}")
    print(f"{'='*60}\n")

    # Train
    final_train_state, training_rewards = train_a2c_jax(config)
    print("\nTraining Summary:")
    print(f"  Final average reward: {np.mean(training_rewards[-10:]):.4f}")
    print(f"  Best average reward: {np.max(training_rewards):.4f}")

    # Auto-evaluate after training completes
    eval_config = config.replace(
        n_sessions=30,
        num_envs=1,
    )
    checkpoint_path = str(Path(f"checkpoints/{config.exp_name}").resolve())

    print(f"\nEvaluating network from checkpoint: {checkpoint_path}\n")
    results, _ = evaluate_a2c_jax(
        config=eval_config,
        checkpoint_path=checkpoint_path,
        save_trajectories=True,
    )

    print(f"\nEvaluation Summary:")
    print(f"  Mean reward rate: {results['mean_reward_rate']:.4f} ± {results['std_reward_rate']:.4f}")

    return results, training_rewards


def main():
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--exp_title', metavar='et', type=str, default='run')
    parser.add_argument('--noise_var', metavar='nv', type=float, default=1e-4)
    parser.add_argument('--activity_reg', metavar='ar', type=float, default=1)
    parser.add_argument('--gamma', metavar='g', type=float, default=0.997)
    parser.add_argument('--env_prediction_weight', metavar='epw', type=float, default=0) # 0.001
    parser.add_argument('--global_reward_weight', metavar='grw', type=float, default=0) # 0.001
    parser.add_argument('--curr_style', metavar='cs', type=str, default='fixed')
    parser.add_argument('--reward_func', metavar='rf', type=str, default='exp')
    parser.add_argument('--agent_type', metavar='at', type=str, default='split')
    parser.add_argument('--rnn_type', metavar='rt', type=str, default='VANILLA')
    parser.add_argument('--seed', metavar='s', type=int, default=0)
    parser.add_argument('--test', action='store_true', help='Run in test/evaluation mode')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for testing')
    parser.add_argument('--test_sessions', type=int, default=30, help='Number of episodes to run in test mode')
    parser.add_argument('--save_trajectories', action='store_true', help='Save trajectory data during testing')
    parser.add_argument('--n_networks', type=int, default=1, help='Number of sequential networks to train (default: 1)')
    parser.add_argument('--intervention_points', type=str, default=None, help='Path to .pkl file containing (N, H) array of points for nearest-neighbour hidden-state intervention (test mode only)')
    args = parser.parse_args()

    """Entry point for training or evaluation"""
    time_stamp = str(datetime.now()).replace(' ', '_')

    # Load config from JSON if provided
    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config_from_json(args.config)
        # Update exp_name with timestamp if not set in config
        if not config.exp_name or config.exp_name == '':
            config = config.replace(exp_name=f"json_config_{time_stamp}")
    else:
        # Build config from command-line arguments (original behavior)
        exp_name = f'{args.exp_title}_seed_{args.seed}_{time_stamp}'

        if args.test:
            # Test/Evaluation mode
            if args.checkpoint_path is None:
                print("Error: --checkpoint_path required for test mode")
                return

            # You can customize the config here
            config = TrainingConfig(
                seed=args.seed,
                exp_name=exp_name,
                n_sessions=args.test_sessions,
                num_envs=1,  # Single env for cleaner episode tracking
                hidden_size=64,
                obs_size=4,
                rnn_type=args.rnn_type if args.rnn_type else 'VANILLA',
                reward_param_style=args.curr_style,
                reward_func_type=args.reward_func,
                unit_noise_std=0,
                input_noise_std=0 #0.02,
            )
        else:
            # Training mode (original behavior)
            config = TrainingConfig(
                seed=args.seed,
                exp_name=exp_name,
                n_sessions=5000,
                num_envs=128,
                learning_rate=1e-4, #1e-4 for GRU, 2e-5, smaller for relu
                entropy_weight=2.5e-3,# for relu 2.5e-5, GRU benefits from larger entropy bonus, like 2.5e-3
                critic_weight=0.05, # 0.5 originally for GRU, 0.04 for relu,
                env_prediction_weight=args.env_prediction_weight, # 0.001,
                global_reward_weight=args.global_reward_weight,
                gamma=args.gamma,
                hidden_size=64, #64
                obs_size=4,
                output_state_save_rate=50,
                rnn_type=args.rnn_type if args.rnn_type else 'VANILLA',
                reward_param_style=args.curr_style,
                reward_func_type=args.reward_func,
                unit_noise_std=0.01,
                input_noise_std=0.01,
            )

    if args.test:
        print("Running in TEST mode")
        print(config)

        config = config.replace(
            n_sessions=args.test_sessions,   # honor --test_sessions (default 30)
            num_envs=1,
        )

        results, trajectories = evaluate_a2c_jax(
            config=config,
            checkpoint_path=os.path.join(os.getcwd(), args.checkpoint_path),
            save_trajectories=args.save_trajectories,
            intervention_points_path=args.intervention_points,
        )

        print(f"\nTest completed! Mean reward: {results['mean_reward_rate']:.4f}")
    else:
        print("Running in TRAINING mode")
        print(f"Training {args.n_networks} network(s)\n")

        all_results = []

        for network_idx in range(args.n_networks):
            # Modify config for this network
            network_seed = config.seed + network_idx
            network_exp_name = f"{config.exp_name}_net{network_idx}" if args.n_networks > 1 else config.exp_name

            network_config = config.replace(
                seed=network_seed,
                exp_name=network_exp_name,
            )

            # Train and evaluate this network
            results, training_rewards = train_and_evaluate_network(network_config)
            all_results.append({
                'network': network_idx,
                'seed': network_seed,
                'exp_name': network_exp_name,
                'eval_results': results,
                'training_rewards': training_rewards,
            })

        # Print summary for all networks
        print(f"\n{'='*60}")
        print("MULTI-NETWORK TRAINING SUMMARY")
        print(f"{'='*60}\n")

        for result in all_results:
            print(f"Network {result['network']} (seed {result['seed']}):")
            print(f"  Mean eval reward rate: {result['eval_results']['mean_reward_rate']:.4f} ± {result['eval_results']['std_reward_rate']:.4f}")
            print(f"  Training: final avg = {np.mean(result['training_rewards'][-10:]):.4f}, best = {np.max(result['training_rewards']):.4f}\n")

        # Final summary - just the eval reward rates
        print(f"\n{'='*60}")
        print("FINAL TEST REWARD RATES")
        print(f"{'='*60}\n")
        for result in all_results:
            print(f"Network {result['network']} (seed {result['seed']}): {result['eval_results']['mean_reward_rate']:.4f} ± {result['eval_results']['std_reward_rate']:.4f}")


if __name__ == "__main__":
    main()

    