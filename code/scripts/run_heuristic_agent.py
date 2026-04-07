import sys
import os
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import jax
import jax.numpy as jnp
from flax import struct
from tqdm.auto import trange
import argparse
import pickle
from datetime import datetime
from agents.n_miss_agent import NMissAgent

# Import your environment
from environments.treadmill_env_jax import (
    TreadmillEnvironment,
    TreadmillEnvParams,
    TreadmillEnvState,
    treadmill_session_default_params
)

# -----------------------------------------------------------
# Arguments
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str, default='heuristic_run')
parser.add_argument('--seed', metavar='s', type=int, default=1)
parser.add_argument('--n_sessions', type=int, default=200, help="Number of episodes")
parser.add_argument('--test_sessions', type=int, default=30)
parser.add_argument('--save_trajectories', action='store_true')
args = parser.parse_args()



# -----------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------
def evaluate_heuristic_agent(
    num_sessions: int,
    num_envs: int,
    save_trajectories: bool = False,
):
    print("Evaluating heuristic agent...")
    rng_key = jax.random.key(args.seed)

    # Initialize environment
    env_params = treadmill_session_default_params()
    env_params = env_params.replace(
        reward_param_style=1,
    )
    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()

    agent = NMissAgent(
        n_envs=num_envs,
        wait_time_for_reward=6,
        odor_cues_indices=[1, 3],
        n_misses=3,
    )

    all_episode_rewards = []
    all_trajectories = [] if save_trajectories else None
    ep_len = 200 * 100

    # Loop over episodes
    for ep in trange(num_sessions, desc="Sessions"):
        rng_key, reset_key = jax.random.split(rng_key)
        reset_keys = jax.random.split(reset_key, num_envs)
        _, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)
        print(env_states)

        episode_rewards = []
        traj_data = [] if save_trajectories else None

        step_count = 0

        for i in range(3):
            print(env_states.reward_params[:, i] / jnp.sum(env_states.reward_params, axis=1) * 6)

        for i in range(ep_len):
            # Decide action based on previous reward (or 0 if first step)
            reward = jnp.zeros((num_envs,)) if step_count == 0 else rewards
            obs = jnp.zeros((num_envs, 4)) if step_count == 0 else obs
            # print(env_states.current_patch.reward_func_param)
            actions = agent.sample_action(
                obs,
                tau=env_states.current_patch.reward_func_param / jnp.sum(env_states.reward_params, axis=1) * 6,
            )

            # Step env
            rng_key, step_key = jax.random.split(rng_key)
            step_keys = jax.random.split(step_key, num_envs)

            obs, env_states, rewards, dones, infos = jax.vmap(
                lambda key, state, act: step_fn(key, state, act, env_params)
            )(step_keys, env_states, actions)

            # Track reward
            episode_rewards.append(rewards)
            agent.append_reward(rewards)

            # Store trajectory if requested
            if save_trajectories:
                traj_data.append({
                    "obs": obs,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                })

            # Update done flags
            step_count += 1

        total_reward = float(jnp.sum(jnp.stack(episode_rewards)))
        all_episode_rewards.append(total_reward)
        print('reward_rate', total_reward / ep_len)

        if save_trajectories:
            all_trajectories.append(traj_data)

    mean_reward = float(jnp.mean(jnp.array(all_episode_rewards)))
    print(f"Mean episode reward: {mean_reward:.4f}")

    results = {
        "episode_rewards": all_episode_rewards,
        "mean_reward": mean_reward,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_dir = Path(f"results/{args.exp_title}")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"heuristic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_file}")

    if save_trajectories:
        traj_file = results_dir / f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(traj_file, "wb") as f:
            pickle.dump(all_trajectories, f)
        print(f"Trajectories saved to {traj_file}")

    return results, all_trajectories


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    results, trajectories = evaluate_heuristic_agent(
        num_sessions=args.n_sessions,
        num_envs=1,  # Run one env per session for clarity
        save_trajectories=args.save_trajectories,
    )


if __name__ == "__main__":
    main()