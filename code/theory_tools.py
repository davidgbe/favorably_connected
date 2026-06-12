"""Theoretical reward-rate calculations for patch-foraging environments."""
import numpy as np
import jax.numpy as jnp
from jax import random


# ---------------------------------------------------------------------------
# Reward probability primitives
# ---------------------------------------------------------------------------

def rewarded_given_i_rewards(i, a, tau):
    return a * np.exp(-i / tau)


def unrewarded_given_i_rewards(i, a, tau):
    return 1 - rewarded_given_i_rewards(i, a, tau)


def p_k_consecutive_rewarded_sites_given_i_rewards(k, i, a, tau):
    return np.prod([rewarded_given_i_rewards(i + r, a, tau) for r in np.arange(k)])


def p_k_additional_rewards_in_n_sites_given_i_rewards(k, n, i, cache, a, tau):
    if n == 0:
        return 0
    elif k == 0:
        return 0
    else:
        if k > n:
            p = 0
        elif k == n:
            p = p_k_consecutive_rewarded_sites_given_i_rewards(k, i, a, tau)
        elif cache[k, n, i] != -1:
            p = cache[k, n, i]
        else:
            p = (
                rewarded_given_i_rewards(i, a, tau)
                * p_k_additional_rewards_in_n_sites_given_i_rewards(k - 1, n - 1, i + 1, cache, a, tau)
                + unrewarded_given_i_rewards(i, a, tau)
                * p_k_additional_rewards_in_n_sites_given_i_rewards(k, n - 1, i, cache, a, tau)
            )
        cache[k, n, i] = p
        return p


def expected_stops_given_tau_and_k_rewards(k, a, tau):
    exp_stops = 0
    max_stop_iter = 500
    cache = np.ones((k + 1, max_stop_iter, k + 1)) * -1
    probs = []
    for n in range(500):
        p_n_stops = p_k_additional_rewards_in_n_sites_given_i_rewards(
            k, n, 0, cache=cache, a=a, tau=tau
        )
        probs.append(p_n_stops)
        exp_stops += p_n_stops * n
    exp_stops += (1 - np.sum(probs)) * max_stop_iter
    return exp_stops, probs


def expected_reward_rate_for_env_and_policy(
    taus, amps, k_rewards_by_tau, d_site_stop, d_site_run, d_intersite, interpatch
):
    total_reward = np.sum(k_rewards_by_tau)
    denom = 0
    for tau, a, k in zip(taus, amps, k_rewards_by_tau):
        if k == 0:
            expected_n = 0
        else:
            expected_n, _ = expected_stops_given_tau_and_k_rewards(k=k, a=a, tau=tau)
        denom += d_intersite + expected_n * (d_site_stop + d_intersite) + d_site_run
    denom += interpatch * len(taus)
    return total_reward / denom


# ---------------------------------------------------------------------------
# Truncated exponential sampler (JAX)
# ---------------------------------------------------------------------------

def sample_truncated_exp(key, bounds, decay_rate):
    """Sample from a truncated exponential distribution via inverse CDF."""
    min_val, max_val = bounds[0], bounds[1]
    u = random.uniform(key)
    exp_max = jnp.exp(-decay_rate * (max_val - min_val))
    sample = -jnp.log(1 - u * (1 - exp_max)) / decay_rate
    return min_val + sample


# ---------------------------------------------------------------------------
# Policy optimisation
# ---------------------------------------------------------------------------

def find_max_policy_for_env(taus, amps, max_stops=10,
                             d_site_stop=6, d_site_run=3,
                             d_intersite=2.25, interpatch=5.28):
    """Brute-force search over integer k-reward policies; returns (max_rate, best_policy)."""
    max_policy = None
    max_rate = 0
    for k_1 in range(max_stops):
        for k_2 in range(max_stops):
            for k_3 in range(max_stops):
                rate = expected_reward_rate_for_env_and_policy(
                    taus=taus,
                    amps=amps,
                    k_rewards_by_tau=(k_1, k_2, k_3),
                    d_site_stop=d_site_stop,
                    d_site_run=d_site_run,
                    d_intersite=d_intersite,
                    interpatch=interpatch,
                )
                if rate > max_rate:
                    max_rate = rate
                    max_policy = (k_1, k_2, k_3)
    return max_rate, max_policy
