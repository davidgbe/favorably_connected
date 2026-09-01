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
RPE_TAU = 100.0          # exponential-filter time constant -> lambda = exp(-1/RPE_TAU)
REWARD_PRED_HIDDEN = 64

# The credit feature at each anchor is the flattened stack of the last N_OBS_TIMESTEPS (obs+reward)
# vectors, so the per-action credit vector represents a short window of recent timesteps, not just
# the current one. This widens the eligibility/credit matrices to (N_OBS_TIMESTEPS*feat_dim, A).
N_OBS_TIMESTEPS = 1
GLOBAL_REWARD_DECAY = 0.999  # Weight for the global reward signal
REWARD_PRED_WEIGHT = 0.1 # was 0.1
# Belief-weighted RPE trace decay lambda: D_t = sum_{l=t'}^{t-1} lambda^(t-l) (r_l - E[r_l]) b_l, where
# t' = the last reward before t. The trace decays by M_DECAY each step and RESETS to 0 at each reward
# (D_{t+1} = keep (D_t + (r_t - E[r_t]) b_t), keep = 0 at a reward else M_DECAY), so it spans only the
# current drought. The additive advantage term is <D_t, b_t>. Same in the loss and the stored trace.
M_DECAY = 0.99
# Value-bootstrap horizon k for the advantage A_t = G_t + <D_t,b_t> + gamma^K V_{t+K} - V_t, with
# G_t = sum_{s=t}^{t+K} gamma^(s-t) r_s. V (critic head) supplies the baseline and the bootstrap;
# requires critic_weight > 0 to train V.
K_BOOT = 20
# (unused in this RPE-trace variant; kept for config/API compatibility with the base reff_pg script)
# Belief head: the critic produces a belief b_t (BELIEF_DIM, unit-norm); a feedforward net predicts the
# next N_BELIEF_PREDICT [obs, reward] tuples from it (belief-prediction MSE, weight BELIEF_PRED_WEIGHT).
# The belief b_t weights past reward-prediction errors in the RPE trace D_t (see M_DECAY): a past
# surprise credits the current step only while the context b still matches (dot product b_l . b_t).
N_BELIEF_PREDICT = 20
BELIEF_DIM = 32
BELIEF_PRED_HIDDEN = 64
BELIEF_PRED_WEIGHT = 1.0


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'n_steps', 'obs_size', 'intervention_fn'])
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
    intervention_fn=None,
) -> Tuple[TrajectoryData, TrainState, TreadmillEnvState]:
    """Collect trajectory using lax.scan over time steps."""

    network = A2CRNNFlax(
        action_size=2,  # Fixed ACTION_SIZE
        hidden_size=hidden_size,
        unit_noise_std=unit_noise_std,
        rnn_type=rnn_type,
        obs_size=obs_size,
        reward_pred_hidden_size=REWARD_PRED_HIDDEN,
        reward_pred_horizon=RPE_H + 1,
        belief_dim=BELIEF_DIM,
        belief_pred_hidden=BELIEF_PRED_HIDDEN,
        belief_pred_horizon=N_BELIEF_PREDICT,
    )

    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()

    step_num = jnp.zeros_like(env_states.exp_filtered_reward_rate)  # (num_envs,)
    n_envs = env_states.exp_filtered_reward_rate.shape[0]
    Dvec0 = jnp.zeros((n_envs, BELIEF_DIM))                         # belief-weighted RPE trace D_t (leaky)

    def scan_step(carry, _):
        train_state, env_states, step_num, Dvec = carry
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

        new_reward_rate = (
            GLOBAL_REWARD_DECAY * new_env_states.exp_filtered_reward_rate 
            + (1 - GLOBAL_REWARD_DECAY) * rewards
        )

        new_env_states = new_env_states.replace(
            exp_filtered_reward_rate=new_reward_rate,
        )

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

        # Belief-weighted RPE trace M = <D_t, b_t>, the additive advantage term. D_t is a leaky vector
        # accumulator of (r_l - E[r_l]) b_l that RESETS at each reward (spans only the current drought):
        # D_{t+1} = keep (D_t + (r_t - E[r_t]) b_t), keep = 0 at a reward else M_DECAY, pre-update carry.
        # E[r_t] is the dedicated 1-step reward head. Matches the loss exactly.
        er0 = reward_pred_1step                                            # E[r_t] from the dedicated 1-step head
        M = jnp.sum(Dvec * belief, axis=-1)                               # <D_t, b_t>  (pre-update carry)
        keep = jnp.where(rewards > 0, 0.0, M_DECAY)[:, None]              # reset AFTER a reward
        Dvec_new = keep * (Dvec + (rewards - er0)[:, None] * belief)      # belief is unit-norm

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
            'M': M,                       # (num_envs,) reward modulation (see above)
            'belief': belief,             # (num_envs, BELIEF_DIM) critic belief vector for S_C context gating
        } | infos

        return (new_train_state, new_env_states, jnp.zeros(rewards.shape[0]), Dvec_new), step_data

    # Run scan over time steps using compile-time constant
    (final_train_state, final_env_states, _, _), trajectory_data = lax.scan(
        scan_step,
        (train_state, env_states, step_num, Dvec0),
        None,
        length=n_steps
    )

    # Reshape trajectory data from (n_steps, num_envs, ...) to (num_envs, n_steps, ...)
    trajectory_data = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectory_data
    )

    trajectory = TrajectoryData(**trajectory_data)

    return trajectory, final_train_state, final_env_states


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
) -> Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], Any, Any]]:
    """Compute A2C loss; collect_trajectory is called here so BPTT flows through the scan."""

    trajectory, final_train_state, final_env_states = collect_trajectory(
        train_state=train_state.replace(params=params),
        env_states=env_states,
        env_params=env_params,
        input_noise_std=input_noise_std,
        unit_noise_std=unit_noise_std,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        obs_size=obs_size,
        n_steps=N_STEPS_PER_UPDATE,
    )

    logits = trajectory.logits            # (B, T, A)rm -rf */reff_pg_exp_gru_initial_prob_offset_v2
    rewards = trajectory.rewards          # (B, T)
    N = N_STEPS_PER_UPDATE                 # chunk length (== T)
    B = rewards.shape[0]                    # num envs
    H = min(RPE_H, N - 1)                  # lag horizon (capped: no target can be > N-1 steps ahead)

    # --- k-step value-bootstrap advantage with an additive belief-weighted RPE trace (note #1) -------
    #   A_t = sum_{s=t}^{t+K} gamma^(s-t) r_s  +  <D_t, b_t>  +  gamma^K V_{t+K}  -  V_t
    # where D_t = sum_{l=t'}^{t-1} lambda^(t-l) (r_l - E[r_l|s_l]) b_l  is a LEAKY vector accumulator of
    # belief-weighted reward-prediction ERRORS (r_l - E[r_l]) that extends back ONLY to t' = the last
    # reward before t (the trace RESETS at each reward, so it spans just the current drought): positive
    # when a reward beats expectation, negative when an expected reward is MISSED. It is carried forward
    # with decay lambda and projected onto the CURRENT belief b_t, so a past surprise credits t only while
    # the context b still matches. The k-step ACTUAL reward is NO LONGER gated by M; instead this additive
    # trace injects the belief-gated surprise straight into A_t. The learned value V supplies the baseline
    # (-V_t) and the bootstrap (gamma^K V_{t+K}).
    #   actor:  -mean_t  A_t * log pi(a_t | s_t)        (A_t stop-gradded)
    #   critic: mean_t ( V_t - stopgrad(G_t + <D_t,b_t> + gamma^K V_{t+K}) )^2   (requires critic_weight>0)
    # E[r] and the belief both come from the CRITIC (the actor reward head was dropped): the dedicated
    # 1-step head gives E[r_t] and the belief b_t is the context. Both are produced inline in
    # collect_trajectory and carried on the trajectory (differentiable here).
    # belief forecast: from each belief b_t, predict the next N_BELIEF_PREDICT [sensory obs, reward]
    # tuples (t..t+NB-1). Produced inline in collect_trajectory (trajectory.belief_pred) so it's
    # differentiable here -- the MSE trains belief_readout + the belief FF net + the critic's recurrent
    # weights. The belief's REWARD channels ARE the reward forecast E[r] (the actor reward head was
    # dropped); its lag-0 channel feeds M. This is the ONLY term shaping the belief; the S_C use is s.g.
    belief_pred = trajectory.belief_pred.reshape(B, N, N_BELIEF_PREDICT, obs_size + 1)  # (B, N, NB, obs+1)
    bj = jnp.clip(jnp.arange(N)[:, None] + jnp.arange(N_BELIEF_PREDICT)[None, :], 0, N - 1)  # (N, NB)  t..t+NB-1
    obs_tgt = lax.stop_gradient(trajectory.observations[:, :, :obs_size])[:, bj]        # (B, N, NB, obs)
    rew_tgt = lax.stop_gradient(rewards)[:, bj][..., None]                              # (B, N, NB, 1)
    belief_target = jnp.concatenate([obs_tgt, rew_tgt], axis=-1)                        # (B, N, NB, obs+1)
    belief_pred_loss = jnp.mean((belief_pred - belief_target) ** 2)

    # dedicated 1-step reward head E[r_t] (critic-side, its own MSE -> a strong, un-swamped reward
    # signal that forces the critic to track within-patch depletion). This is E[r] for M.
    reward_pred_1step = trajectory.reward_pred_1step                      # (B, N)  E[r_t]
    reward_pred_1step_loss = jnp.mean((reward_pred_1step - lax.stop_gradient(rewards)) ** 2)

    rp_sg = lax.stop_gradient(reward_pred_1step)                         # (B, N)  E[r_t]
    rewards_sg = lax.stop_gradient(rewards)

    # Belief-weighted RPE trace (note #1): D_t = sum_{l=t'}^{t-1} lambda^(t-l) (r_l - E[r_l]) b_l, a LEAKY
    # vector accumulator of belief-weighted reward-prediction errors that extends back ONLY to t' = the
    # last reward before t (RESET at each reward). It thus spans only the CURRENT drought since the last
    # reward: post-reward failures (RPE = -E[r] < 0) accumulate, and the trace zeroes out when the next
    # reward lands. Emitted as <D_t, b_t> (the additive advantage term); recursion
    # D_{t+1} = keep_t (D_t + (r_t - E[r_t]) b_t) with keep_t = 0 at a reward else lambda; D_0 = 0; the
    # scan emits the PRE-update carry. b_hat/E[r] are stop-gradded (the belief is shaped only by its
    # prediction MSEs, not the advantage). O(N) single scan.
    rpe       = rewards_sg - rp_sg                                      # (B, N)  r_l - E[r_l]
    is_reward = rewards_sg > 0                                          # (B, N)  reward step -> reset the trace
    b_hat = lax.stop_gradient(trajectory.belief)                        # (B, N, BELIEF_DIM)  unit-norm
    def _rpe_scan(carry, xt):
        D = carry                                                       # (B, BELIEF_DIM) = D_t (sum over t' <= l < t)
        rpe_k, bh, isr = xt
        A_extra_t = jnp.sum(D * bh, axis=-1)                            # <D_t, b_t>  (pre-update carry)
        keep = jnp.where(isr, 0.0, M_DECAY)                            # (B,)  reset AFTER a reward (else decay lambda)
        D_new = keep[:, None] * (D + rpe_k[:, None] * bh)              # D_{t+1} = keep (D_t + (r_t-E[r_t]) b_t)
        return D_new, A_extra_t
    _, M = lax.scan(_rpe_scan, jnp.zeros((B, BELIEF_DIM)),
                    (jnp.swapaxes(rpe, 0, 1), jnp.swapaxes(b_hat, 0, 1), jnp.swapaxes(is_reward, 0, 1)))
    M = jnp.swapaxes(M, 0, 1)                                            # (B, N)  <D_t, b_t> (stored/plotted as "M")

    # k-step value-bootstrap advantage (see note #1). G_t = sum_{i=0}^{K} gamma^i r_{t+i}; near the
    # chunk end the window and bootstrap shrink to the actual remaining horizon `gap` so they stay
    # consistent. The belief-weighted RPE trace <D_t,b_t> (=M) is ADDED to G_t (no longer a gate); V
    # provides the -V_t baseline and the gamma^gap V_{t+gap} bootstrap.
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
    td_target = G + 3 * lax.stop_gradient(M) + (gamma ** gap)[None, :] * v_boot   # (B, N)  G_t + <D_t,b_t> + gamma^K V_{t+K}
    critic_loss = jnp.mean((V - lax.stop_gradient(td_target)) ** 2)    # fit V to the TD target
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

    return total_loss, (metrics, jax.lax.stop_gradient(final_train_state), jax.lax.stop_gradient(final_env_states))


def compute_n_step_returns(rewards, gamma, v_t, reverse=True):
    """
    Compute n-step returns for n=0 to max_n efficiently
    
    Args:
        rewards: (batch_size, time_steps) - rewards at each timestep
        gamma: discount factor
        
    Returns:
        n_step_returns: (batch_size, time_steps, max_n+1) where [:, :, n] contains n-step returns
    """

    def compute_return(carry, reward):
        (i, rolling_sum) = carry
        rolling_sum = reward + gamma * rolling_sum
        return (i+1, rolling_sum), rolling_sum
    
    _, returns = lax.scan(
        compute_return,
        (0, v_t),
        rewards,
        reverse=reverse,
    )
    return returns


def nearest_future_reward(rewards, decay):
    """For each t, decay^(k-t) * r_k where k >= t is the CLOSEST future step with a reward (r_k > 0).

    Only the nearest reward is used (not the discounted sum of all future rewards). Reverse recurrence:
        v_t = r_t                if r_t > 0   (the reward is at t: decay^0 * r_t)
        v_t = decay * v_{t+1}    otherwise    (carry the nearest future reward one step closer)
    with v beyond the block == 0.

    Args:
        rewards: (time_steps,) per-step rewards for one env.
        decay:   per-step discount.
    Returns:
        (time_steps,) nearest-future-reward values.
    """
    def step(carry, r):
        v = jnp.where(r > 0.0, r, decay * carry)
        return v, v

    _, out = lax.scan(step, 0.0, rewards, reverse=True)
    return out


def forward_value_targets(rewards, lam, anchor):
    """Forward discounted accumulation of value from a chunk-start anchor.

        target[0] = anchor
        target[t] = lam * target[t-1] + rewards[t-1]
                  = lam^t * anchor + sum_{s=0}^{t-1} lam^{t-1-s} rewards[s]

    Args:
        rewards: (time_steps,) per-step rewards for one env.
        lam: discount.
        anchor: scalar value v[0,0] used as the t=0 target.
    Returns:
        (time_steps,) forward-accumulated value targets (target[0] == anchor).
    """
    def step(carry, r):
        out = carry                 # emit the accumulated value BEFORE this step's reward
        new = lam * carry + r       # advance the accumulator for the next step
        return new, out

    _, targets = lax.scan(step, anchor, rewards)
    return targets


def compute_gaes(rewards, values, gamma, lam):
    """
    rewards: [B, T]
    values:  [B, T+1]
    --> returns advantages: [B, T]
    """

    # Move time to axis 0, because scan iterates over axis 0
    rewards_t = rewards.T            # [T, B]
    values_t  = values.T             # [T+1, B]

    def gae_scan(carry, x_t):
        reward_t, value_t, value_tp1 = x_t
        delta = reward_t + gamma * value_tp1 - value_t
        gae = delta + gamma * lam * carry
        return gae, gae

    # xs is a tuple of time-major sequences
    xs = (rewards_t[:-1], values_t[:-1], values_t[1:])   # shapes all [T, B]

    # reverse=True makes scan go from T-1 → 0
    _, adv_t = jax.lax.scan(
        gae_scan,
        init=jnp.zeros(rewards.shape[0]),  # [B]
        xs=xs,
        reverse=True
    )

    return jnp.concatenate((
        adv_t.T,
        jnp.zeros((rewards.shape[0], 1))
    ), axis=1)


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'obs_size'])
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
) -> Tuple[TrainState, TreadmillEnvState, Dict[str, jnp.ndarray]]:
    """Single training step"""
    
    grad_fn = jax.grad(compute_a2c_loss, has_aux=True)
    grads, (metrics, final_train_state, final_env_states) = grad_fn(
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
    
    return final_train_state, final_env_states, metrics

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
    rnn_type: str = 'GRU'
    init_scale: float = 1.0

    # Training params (runtime configurable)
    seed: int = 0
    n_sessions: int = 5000

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


@partial(jax.jit, static_argnames=['action_size', 'hidden_size', 'unit_noise_std', 'rnn_type', 'obs_size'])
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
) -> Tuple[TrainState, TreadmillEnvState, Dict[str, jnp.ndarray]]:
    """Run all training updates with full metrics collection"""
    
    def update_step(carry, _):
        train_state, env_states = carry
        
        new_train_state, new_env_states, metrics = train_step(
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
        )
        
        return (new_train_state, new_env_states), metrics
    
    # Run scan over all updates
    (final_train_state, final_env_states), all_metrics = lax.scan(
        update_step,
        (train_state, env_states),
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
        reward_pred_hidden_size=REWARD_PRED_HIDDEN,
        window_dim=RPE_K * (config.obs_size + 1) + config.action_size + 1,
        reward_pred_horizon=RPE_H + 1,
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
            trajectory, _, _ = collect_trajectory(
                train_state=save_train_state,
                env_states=save_env_states,
                env_params=env_params,
                input_noise_std=config.input_noise_std,
                unit_noise_std=config.unit_noise_std,
                rnn_type=config.rnn_type,
                hidden_size=config.hidden_size,
                obs_size=config.obs_size,
                n_steps=N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE,
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
        reward_pred_hidden_size=REWARD_PRED_HIDDEN,
        window_dim=RPE_K * (config.obs_size + 1) + config.action_size + 1,
        reward_pred_horizon=RPE_H + 1,
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
        trajectory, final_train_state, final_env_states = collect_trajectory(
            train_state=train_state,
            env_states=env_states,
            env_params=env_params,
            input_noise_std=0,  # No noise during evaluation
            unit_noise_std=0,
            rnn_type=config.rnn_type,
            hidden_size=config.hidden_size,
            obs_size=config.obs_size,
            n_steps=session_steps,
            intervention_fn=intervention_fn,
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

    