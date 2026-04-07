import sys
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))


import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax import linen as nn
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
import cma
from typing import Any, Dict, Tuple
import argparse
from tqdm.auto import tqdm
import pickle
import optax
from agents.a2c_rnn_flax import A2CRNNFlax


# Import from your original script - adjust path as needed
from train_treadmill_agent_jax import (
    create_train_state,
    run_session_updates_with_metrics,
    TrainingConfig,
    TreadmillEnvironment,
    TrainState,
    treadmill_session_default_params,
    N_UPDATES_PER_SESSION,
)

# Parse arguments
parser = argparse.ArgumentParser(description='Meta-learning weight initialization with CMA-ES')
parser.add_argument('--init_network_hidden', type=int, default=12, help='Hidden size for initialization network')
parser.add_argument('--cma_sigma', type=float, default=1, help='Initial standard deviation for CMA-ES')
parser.add_argument('--cma_popsize', type=int, default=16, help='Population size for CMA-ES')
parser.add_argument('--cma_generations', type=int, default=3000, help='Number of CMA-ES generations')
parser.add_argument('--eval_sessions', type=int, default=15, help='Sessions to train each candidate')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--save_dir', type=str, default='meta_learning_results', help='Directory to save results')
parser.add_argument('--exp_name', type=str, default='cmaes_weight_init', help='Experiment name')
args = parser.parse_args()


class WeightInitNetwork(nn.Module):
    """Small feedforward network to initialize RNN weights based on position indices"""
    hidden_size: int
    
    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size, use_bias=True)
        self.output = nn.Dense(1, use_bias=True)
        
    def __call__(self, indices):
        """
        Args:
            indices: (batch_size, 2) - row and column indices normalized to [0,1]
        Returns:
            weights: (batch_size, 1) - initialized weight values
        """
        x = self.dense1(indices)
        x = nn.tanh(x)
        x = self.output(x)
        return x.squeeze(-1)


def generate_weight_indices(weight_shape):
    """Generate normalized row, column indices for a weight matrix"""
    rows, cols = weight_shape
    row_indices = jnp.arange(rows)[:, None] / max(rows - 1, 1)
    col_indices = jnp.arange(cols)[None, :] / max(cols - 1, 1)
    
    # Broadcast to create all combinations
    row_grid = jnp.broadcast_to(row_indices, (rows, cols))
    col_grid = jnp.broadcast_to(col_indices, (rows, cols))
    
    # Stack and reshape to (rows*cols, 2)
    indices = jnp.stack([row_grid.flatten(), col_grid.flatten()], axis=1)
    return indices


def initialize_weights_with_network(init_network_params, weight_shape, rng_key):
    """Initialize a weight matrix using the small network"""
    init_network = WeightInitNetwork(hidden_size=args.init_network_hidden)
    
    # Generate position indices
    indices = generate_weight_indices(weight_shape)
    
    # Get weights from initialization network
    weights = init_network.apply(init_network_params, indices)
    
    # Reshape back to original weight shape
    weights = weights.reshape(weight_shape)
    
    return weights


def apply_weight_init_network(rnn_params, init_network_params, rng_key):
    """Apply the weight initialization network to all RNN weight matrices"""
    flat_params = flatten_dict(rnn_params)
    new_flat_params = {}

    def valid_key_check(key):
        return 'kernel' in key[-1] and ('rnn_actor' in key[1] or 'rnn_critic' in key[1]) and ('hidden_projection' in key[2])
    
    for key, param in flat_params.items():
        if valid_key_check(key) and len(param.shape) == 2:  # Only 2D weight matrices
            new_flat_params[key] = initialize_weights_with_network(
                init_network_params, param.shape, rng_key
            )
        else:
            new_flat_params[key] = param  # Keep biases and other params unchanged
    
    return unflatten_dict(new_flat_params)


def create_init_network_params(rng_key, hidden_size):
    """Create parameters for the weight initialization network"""
    init_network = WeightInitNetwork(hidden_size=hidden_size)
    dummy_input = jnp.zeros((1, 2))  # Row, col indices
    params = init_network.init(rng_key, dummy_input)
    return params


def params_to_vector(params):
    """Convert parameter pytree to flat vector for CMA-ES"""
    flat_params = flatten_dict(params)
    vectors = []
    for key in sorted(flat_params.keys()):
        vectors.append(flat_params[key].flatten())
    return jnp.concatenate(vectors)


def vector_to_params(vector, template_params):
    """Convert flat vector back to parameter pytree"""
    flat_template = flatten_dict(template_params)
    new_flat_params = {}
    
    start_idx = 0
    for key in sorted(flat_template.keys()):
        param_shape = flat_template[key].shape
        param_size = np.prod(param_shape)
        
        param_values = vector[start_idx:start_idx + param_size]
        new_flat_params[key] = param_values.reshape(param_shape)
        start_idx += param_size
    
    return unflatten_dict(new_flat_params)


def modified_create_train_state(
    rng_key,
    config: TrainingConfig,
    init_network_params: Any = None,
):
    
    # Network input size: obs + prev_obs + prev_action + prev_reward
    input_size = config.obs_size + config.action_size + 1
    
    # Initialize network
    network = A2CRNNFlax(
        action_size=config.action_size,
        hidden_size=config.hidden_size, 
        rnn_type=config.rnn_type,
        var_noise=config.var_noise
    )
    
    # Initialize parameters
    rng_key, param_key = random.split(rng_key)
    dummy_input = jnp.zeros((1, input_size))
    dummy_hidden = jnp.zeros((1, config.hidden_size))
    
    params = network.init(
        param_key, 
        dummy_input, 
        dummy_hidden, 
        dummy_hidden
    )

    # Apply weight initialization network if provided
    if init_network_params is not None:
        rng_key, init_key = random.split(rng_key)
        params = apply_weight_init_network(params, init_network_params, init_key)

    # Create optimizer and other components
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize hidden states and previous step info
    actor_hidden = jnp.zeros((config.num_envs, config.hidden_size))
    critic_hidden = jnp.zeros((config.num_envs, config.hidden_size))
    prev_obs = jnp.zeros((config.num_envs, config.obs_size))
    prev_action = jnp.zeros((config.num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((config.num_envs,))
    
    return TrainState(
        params=params,
        opt_state=opt_state,
        rng_key=rng_key,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        prev_obs=prev_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        learning_rate=config.learning_rate,
        grads=None,
    )


def evaluate_init_network(init_network_params, config: TrainingConfig, rng_key):
    """Evaluate a weight initialization network by training an agent for a few sessions"""
    
    try:
        # Create training state with the initialization network
        train_state = modified_create_train_state(
            rng_key=rng_key,
            config=config,
            init_network_params=init_network_params,
        )
        
        # Initialize environments
        reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()
        env_params = treadmill_session_default_params()
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        
        _, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)
        
        session_reward_rates = None
        
        # Train for specified number of sessions
        for session in range(args.eval_sessions):
            train_state, env_states, all_metrics = run_session_updates_with_metrics(
                train_state=train_state,
                env_states=env_states, 
                env_params=env_params,
                gamma=config.gamma,
                critic_weight=config.critic_weight,
                entropy_weight=config.entropy_weight,
                input_noise_std=config.input_noise_std,
                action_size=config.action_size,
                hidden_size=config.hidden_size,
                var_noise=config.var_noise,
                rnn_type=config.rnn_type,
            )
            
            # Collect loss from this session
            session_reward_rates = all_metrics['mean_reward'] 
        
        # Return negative average loss (CMA-ES minimizes, but we want to minimize loss)
        return -np.mean(session_reward_rates.tolist())
        
    except Exception as e:
        print(f"Error evaluating network: {e}")
        return 1e6  # Return large penalty for failed evaluations


def objective_function(vector, template_params, config, rng_key):
    """Objective function for CMA-ES optimization"""
    
    # Convert vector back to parameters
    init_network_params = vector_to_params(jnp.array(vector), template_params)
    # for k, v in flatten_dict(init_network_params['params']).items():
    #     print(k, v.shape)
    
    # Evaluate the initialization network
    fitness = evaluate_init_network(init_network_params, config, rng_key)
    
    return fitness


def run_cmaes_optimization():
    """Main function to run CMA-ES optimization of weight initialization network"""
    
    print("Starting CMA-ES optimization of weight initialization network")
    print(f"Population size: {args.cma_popsize}")
    print(f"Generations: {args.cma_generations}")
    print(f"Evaluation sessions per candidate: {args.eval_sessions}")
    
    # Setup
    rng_key = random.PRNGKey(args.seed)
    config = TrainingConfig(
        n_sessions=args.eval_sessions,  # Use eval_sessions for short training
        num_envs=32,  # Smaller for faster evaluation
        hidden_size=64,  # Smaller for faster evaluation
        rnn_type='VANILLA',
        learning_rate=1e-4,
    )
    
    # Create template parameters for the initialization network
    rng_key, template_key = random.split(rng_key)
    template_params = create_init_network_params(template_key, args.init_network_hidden)
    
    # Convert to vector to get dimensionality
    init_vector = params_to_vector(template_params)
    dimension = len(init_vector)
    
    print(f"Initialization network parameter dimension: {dimension}")
    
    # Setup CMA-ES
    initial_mean = np.array(init_vector)  # Start from random initialization
    cma_opts = {
        'popsize': args.cma_popsize,
        'seed': args.seed,
        'verbose': 1,
    }
    
    es = cma.CMAEvolutionStrategy(initial_mean, args.cma_sigma, cma_opts)
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimization loop
    generation = 0
    best_fitness = float('inf')
    best_params = None
    
    fitness_history = []
    
    try:
        while not es.stop() and generation < args.cma_generations:
            print(f"\nGeneration {generation}")
            
            # Get candidate solutions
            solutions = es.ask()
            
            # Evaluate each solution
            fitness_values = []
            
            for i, solution in enumerate(solutions):
                rng_key, eval_key = random.split(rng_key)
                
                print(f"  Evaluating candidate {i+1}/{len(solutions)}")
                fitness = objective_function(solution, template_params, config, eval_key)
                fitness_values.append(fitness)
                
                # Track best
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_params = vector_to_params(jnp.array(solution), template_params)
                    print(f"    New best fitness: {best_fitness:.6f}")
            
            # Tell CMA-ES the fitness values
            es.tell(solutions, fitness_values)
            
            # Log generation results
            gen_best = min(fitness_values)
            gen_mean = np.mean(fitness_values)
            gen_std = np.std(fitness_values)
            
            print(f"Generation {generation}: best={gen_best:.6f}, mean={gen_mean:.6f}, std={gen_std:.6f}")
            
            fitness_history.append({
                'generation': generation,
                'best': gen_best,
                'mean': gen_mean,
                'std': gen_std,
                'all_fitness': fitness_values.copy()
            })
            
            # Save intermediate results
            if generation % 5 == 0:
                results = {
                    'fitness_history': fitness_history,
                    'best_fitness': best_fitness,
                    'best_params': best_params,
                    'config': config,
                    'args': vars(args),
                    'generation': generation,
                }
                
                with open(save_dir / f'results_gen_{generation}.pkl', 'wb') as f:
                    pickle.dump(results, f)
            
            generation += 1
    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    # Save final results
    final_results = {
        'fitness_history': fitness_history,
        'best_fitness': best_fitness,
        'best_params': best_params,
        'config': config,
        'args': vars(args),
        'final_generation': generation,
    }
    
    with open(save_dir / 'final_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\nOptimization completed!")
    print(f"Best fitness achieved: {best_fitness:.6f}")
    print(f"Results saved to: {save_dir}")
    
    return final_results


def load_and_test_best_params(results_path: str, config: TrainingConfig = None):
    """Load best parameters and test them with a full training run"""
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    best_params = results['best_params']
    print(f"Loaded best params with fitness: {results['best_fitness']:.6f}")
    
    if config is None:
        # Create a full training config for testing
        config = TrainingConfig(
            n_sessions=100,  # Longer training for testing
            num_envs=64,
            hidden_size=128,
        )
    
    # Run a full training session with the best initialization
    rng_key = random.PRNGKey(42)  # Fixed seed for reproducibility
    
    print("Testing best initialization network with full training...")
    fitness = evaluate_init_network(best_params, config, rng_key)
    print(f"Full training fitness: {fitness:.6f}")
    
    return fitness


if __name__ == "__main__":
    # Run the optimization
    results = run_cmaes_optimization()
    
    # Optionally test the best result
    print("\nTesting best result with longer training...")
    test_config = TrainingConfig(
        n_sessions=5,
        num_envs=64, 
        hidden_size=128,
    )
    load_and_test_best_params(
        Path(args.save_dir) / args.exp_name / 'final_results.pkl',
        test_config
    )