"""Shared plotting and analysis functions for GRU dynamics notebooks."""
import os
import pickle
from copy import copy
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import seaborn as sns
import jax
import jax.numpy as jnp
import optax
from jax import jit
from flax.training import checkpoints
from pathlib import Path
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from aux_funcs import format_plot, format_pc_plot
from nb_analysis_tools import load_trajectory_data, parse_behavioral_data


# --- (from cell 7d8c2244) ---
"""
Fixed Point Analysis for RNN Dynamics

This script finds and analyzes fixed points in trained RNN agents.
A fixed point h* satisfies: h* = f(h*, x) for a given input x.
"""

from typing import List, Tuple
from agents.a2c_rnn_flax import init_network_and_params


def rnn_step(hidden_state, input_vec, params, network):
    """
    Execute one RNN step using the actual network.
    
    Args:
        hidden_state: Current hidden state (hidden_size,)
        input_vec: Input vector (input_size,)
        params: Network parameters
        network: A2CRNNFlax network instance
    
    Returns:
        Next hidden state (hidden_size,)
    """
    # Add batch dimension for network compatibility
    h_batch = hidden_state[None, :]
    x_batch = input_vec[None, :]
    
    # Forward pass through RNN using apply with method argument
    h_next_batch = network.apply(
        params, h_batch, x_batch, 
        method=lambda module, h, x: module.rnn_actor(h, x)
    )
    
    return h_next_batch[0].squeeze()


def find_fixed_point(
    params,
    network,
    input_vec,
    h_init,
    max_steps=5000,
    learning_rate=0.01,
    tolerance=1e-6,
    verbose=False
) -> Tuple[jnp.ndarray, bool]:
    """
    Find a fixed point via gradient descent on ||h - f(h, x)||^2.
    
    Args:
        params: Network parameters
        network: A2CRNNFlax instance
        input_vec: Fixed input context
        h_init: Initial hidden state guess
        max_steps: Maximum optimization steps
        learning_rate: Adam learning rate
        tolerance: Convergence threshold
        verbose: Print progress
    
    Returns:
        (fixed_point, converged): Fixed point and convergence flag
    """
    def loss_fn(h):
        h_next = rnn_step(h, input_vec, params, network)
        return jnp.sum((h - h_next.squeeze()) ** 2)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(h_init)
    h = h_init
    
    @jit
    def update(h, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(h)
        updates, opt_state = optimizer.update(grads, opt_state)
        h = optax.apply_updates(h, updates)
        return h, opt_state, loss
    
    for step in range(max_steps):
        h, opt_state, loss = update(h, opt_state)
        
        if verbose and step % 500 == 0:
            print(f"  Step {step:4d}, Loss: {loss:.8f}")
        
        if loss < tolerance:
            if verbose:
                print(f"  ✓ Converged at step {step} (loss: {loss:.8f})")
            return h, True
    
    if verbose:
        print(f"  ✗ Did not converge (final loss: {loss:.8f})")
    return h, False


def find_multiple_fixed_points(
    params,
    network,
    input_vec,
    n_attempts=20,
    hidden_dim=64,
    uniqueness_threshold=0.1,
    **kwargs
) -> List[jnp.ndarray]:
    """
    Search for multiple fixed points using random initializations.
    
    Args:
        params: Network parameters
        network: A2CRNNFlax instance
        input_vec: Fixed input context
        n_attempts: Number of random initializations
        hidden_dim: Hidden state dimension
        uniqueness_threshold: Distance threshold for duplicate detection
        **kwargs: Additional arguments for find_fixed_point
    
    Returns:
        List of unique fixed points found
    """
    fixed_points = []
    
    print(f"Searching for fixed points ({n_attempts} attempts)...\n")
    
    for i in range(n_attempts):
        # print(f"Attempt {i+1}/{n_attempts}")
        
        # Random initialization
        key = jax.random.PRNGKey(i)
        index = jax.random.randint(key, 1, 0, traj_data['actor_hidden'].shape[0])
        h_init = traj_data['actor_hidden'][index, :].squeeze()
        
        h_star, converged = find_fixed_point(
            params, network, input_vec, h_init, **kwargs
        )
        
        if converged:
            # Check uniqueness
            is_unique = all(
                jnp.linalg.norm(h_star - fp) >= uniqueness_threshold
                for fp in fixed_points
            )
            
            if is_unique:
                fixed_points.append(h_star)
                # print(f"  → New fixed point found! (Total: {len(fixed_points)})\n")
            else:
                pass
                # print(f"  → Duplicate (matches existing fixed point)\n")
    
    return fixed_points


def analyze_stability(fixed_points, input_vec, params, network):
    """
    Analyze stability of fixed points via eigenvalues of Jacobian.
    
    Args:
        fixed_points: List of fixed point states
        input_vec: Input context used to find fixed points
        params: Network parameters
        network: A2CRNNFlax instance
    
    Returns:
        List of dicts with stability info
    """
    print("\n" + "="*60)
    print("STABILITY ANALYSIS")
    print("="*60)
    
    results = []
    
    for i, fp in enumerate(fixed_points):
        # Compute Jacobian at fixed point
        jacobian = jax.jacfwd(lambda h: rnn_step(h, input_vec, params, network))(fp)
        eigenvalues = jnp.linalg.eigvals(jacobian)
        max_eig = jnp.max(jnp.abs(eigenvalues))
        
        is_stable = max_eig < 1.0
        stability_str = "Stable (attractor)" if is_stable else "Unstable (saddle)"
        
        result = {
            'fixed_point': fp,
            'eigenvalues': eigenvalues,
            'max_eigenvalue': max_eig,
            'is_stable': is_stable
        }
        results.append(result)
        
        print(f"\nFixed Point {i+1}:")
        print(f"  Max |eigenvalue|: {max_eig:.6f}")
        print(f"  Status: {stability_str}")
        print(f"  Norm: {jnp.linalg.norm(fp):.4f}")
    
    return results


def visualize_fixed_points(fixed_points, stability_results=None):
    """
    Visualize fixed points in 2D projection.
    
    Args:
        fixed_points: List of fixed point states
        stability_results: Optional stability analysis results
    """
    if len(fixed_points) == 0:
        print("No fixed points to visualize")
        return
    
    fp_array = jnp.array(fixed_points)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: First 2 dimensions
    colors = ['green' if r['is_stable'] else 'red' 
              for r in stability_results] if stability_results else None
    
    ax1.scatter(fp_array[:, 0], fp_array[:, 1], 
                s=200, c=colors or range(len(fixed_points)),
                cmap='viridis', edgecolors='black', linewidths=2,
                alpha=0.7)
    
    for i, fp in enumerate(fp_array):
        ax1.annotate(f'{i+1}', (fp[0], fp[1]), 
                    fontsize=12, ha='center', va='center')
    
    ax1.set_xlabel('Hidden Dimension 1', fontsize=12)
    ax1.set_ylabel('Hidden Dimension 2', fontsize=12)
    ax1.set_title('Fixed Points (First 2 Dimensions)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Eigenvalue spectrum
    if stability_results:
        for i, result in enumerate(stability_results):
            eigs = result['eigenvalues']
            color = 'green' if result['is_stable'] else 'red'
            ax2.scatter(eigs.real, eigs.imag, 
                       s=30, c=color, alpha=0.6, label=f'FP {i+1}')
        
        # Unit circle
        theta = jnp.linspace(0, 2*jnp.pi, 100)
        ax2.plot(jnp.cos(theta), jnp.sin(theta), 'k--', 
                linewidth=2, label='Unit circle', alpha=0.5)
        
        ax2.set_xlabel('Real Part', fontsize=12)
        ax2.set_ylabel('Imaginary Part', fontsize=12)
        ax2.set_title('Eigenvalue Spectrum', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.axvline(x=0, color='k', linewidth=0.5)
        ax2.legend(fontsize=10)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

def find_and_plot_fixed_points(input_vec, n_attempts=100):
    # Define input context
    # Input format: [obs(4), prev_action_onehot(2), prev_reward(1)]
    input_dim = CONFIG['obs_size'] + CONFIG['action_size'] + 1
    
    # Or try specific contexts:
    # sample_input = jnp.array([1, 0, 0, 0,  # obs (one-hot)
    #                           1, 0,         # prev_action (one-hot)
    #                           0.5,          # prev_reward
    #                           0.8])         # env_quality
    
    print(f"Input dimension: {input_dim}")
    
    # Find fixed points
    fixed_points = find_multiple_fixed_points(
        params=params,
        network=network,
        input_vec=input_vec,
        n_attempts=n_attempts,
        hidden_dim=CONFIG['hidden_size'],
        max_steps=10000,
        learning_rate=0.01,
        tolerance=1e-6,
        verbose=False
    )
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found {len(fixed_points)} unique fixed points")
    print(f"{'='*60}")
    
    # Analyze stability
    if fixed_points:
        stability_results = analyze_stability(
            fixed_points, sample_input, params, network
        )
        
        # Visualize
        visualize_fixed_points(fixed_points, stability_results)
    else:
        print("\nNo fixed points found. Try:")
        print("  - Different input contexts")
        print("  - More random initializations")
        print("  - Different learning rates or tolerances")
    return fixed_points


# --- (from cell 197189d8) ---
def rnn_step_batch(hidden_states, input_vec, params, network):
    """
    Execute one RNN step for a batch of hidden states.
    
    Args:
        hidden_states: Batch of hidden states (batch_size, hidden_size)
        input_vec: Input vector (input_size,) - same for all
        params: Network parameters
        network: A2CRNNFlax network instance
    
    Returns:
        Next hidden states (batch_size, hidden_size)
    """
    batch_size = hidden_states.shape[0]
    # Repeat input for batch
    x_batch = jnp.tile(input_vec[None, :], (batch_size, 1))
    
    # Forward pass through RNN
    h_next_batch = network.apply(
        params, hidden_states, x_batch, 
        method=lambda module, h, x: module.rnn_actor(h, x)
    )
    
    return h_next_batch[0].squeeze()


def find_fixed_points_batch(
    params,
    network,
    input_vec,
    h_inits,
    max_steps=5000,
    learning_rate=0.01,
    tolerance=1e-6,
    verbose=False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find fixed points for multiple initializations in parallel.
    
    Args:
        params: Network parameters
        network: A2CRNNFlax instance
        input_vec: Fixed input context
        h_inits: Initial hidden states (batch_size, hidden_size)
        max_steps: Maximum optimization steps
        learning_rate: Adam learning rate
        tolerance: Convergence threshold
        verbose: Print progress
    
    Returns:
        (fixed_points, converged): Arrays of shape (batch_size, hidden_size) and (batch_size,)
    """
    batch_size = h_inits.shape[0]
    
    # Define loss function for a single hidden state
    def loss_fn_single(h):
        h_next = rnn_step_batch(h[None, :], input_vec, params, network).squeeze()
        return jnp.sum((h - h_next) ** 2)
    
    # Vectorize loss and gradient computation across batch
    loss_and_grad_fn = jax.vmap(jax.value_and_grad(loss_fn_single))
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(h_inits)
    h_batch = h_inits
    
    @jit
    def update(h_batch, opt_state):
        losses, grads = loss_and_grad_fn(h_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        h_batch = optax.apply_updates(h_batch, updates)
        return h_batch, opt_state, losses
    
    converged = jnp.zeros(batch_size, dtype=bool)
    
    for step in range(max_steps):
        h_batch, opt_state, losses = update(h_batch, opt_state)
        
        # Check convergence for each sample
        newly_converged = (losses < tolerance) & (~converged)
        converged = converged | newly_converged
        
        if verbose and step % 20000 == 0:
            n_converged = jnp.sum(converged)
            print(f"  Step {step:4d}, Converged: {n_converged}/{batch_size}, Mean loss: {jnp.mean(losses):.8f}")
        
        if jnp.all(converged):
            if verbose:
                print(f"  ✓ All converged at step {step}")
            break
    
    if verbose:
        n_converged = jnp.sum(converged)
        print(f"  Final: {n_converged}/{batch_size} converged")
    
    return h_batch, converged, losses


def filter_unique_fixed_points(
    fixed_points,
    converged,
    uniqueness_threshold=0.1
):
    """
    Filter out duplicate fixed points.
    
    Args:
        fixed_points: Array of fixed points (batch_size, hidden_size)
        converged: Boolean array indicating convergence (batch_size,)
        uniqueness_threshold: Distance threshold for duplicates
    
    Returns:
        Array of unique fixed points
    """
    # Only consider converged points
    fps = fixed_points[converged]
    
    if len(fps) == 0:
        return jnp.array([])
    
    unique_fps = [fps[0]]
    
    for fp in fps[1:]:
        is_unique = all(
            jnp.linalg.norm(fp - ufp) >= uniqueness_threshold
            for ufp in unique_fps
        )
        if is_unique:
            unique_fps.append(fp)
    
    return jnp.array(unique_fps)


def analyze_stability_batch(fixed_points, input_vec, params, network):
    """
    Analyze stability of fixed points via eigenvalues of Jacobian.
    
    Args:
        fixed_points: Array of fixed points (n_points, hidden_size)
        input_vec: Input context
        params: Network parameters
        network: A2CRNNFlax instance
    
    Returns:
        List of eigenvalue arrays
    """
    def compute_jacobian_eigs(fp):
        jacobian = jax.jacfwd(
            lambda h: rnn_step_batch(h[None, :], input_vec, params, network).squeeze()
        )(fp)
        return jnp.linalg.eigvals(jacobian)
    
    # Vectorize over fixed points
    all_eigenvalues = jax.vmap(compute_jacobian_eigs)(fixed_points)
    return all_eigenvalues


def plot_eigenspectra_grid(eigenvalues_list, n_plots=9):
    """
    Plot eigenspectra in a 3x3 grid.
    
    Args:
        eigenvalues_list: Array of eigenvalues (n_points, hidden_size)
        n_plots: Number of plots (default 9 for 3x3 grid)
    """
    n_fps = min(len(eigenvalues_list), n_plots)
    
    if n_fps == 0:
        print("No fixed points to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    axes = axes.flatten()
    
    # Unit circle
    theta = jnp.linspace(0, 2*jnp.pi, 100)
    unit_circle_x = jnp.cos(theta)
    unit_circle_y = jnp.sin(theta)
    
    for i in range(9):
        ax = axes[i]
        
        if i < n_fps:
            eigs = eigenvalues_list[i]
            max_eig = jnp.max(jnp.abs(eigs))
            is_stable = max_eig < 1.0
            color = 'green' if is_stable else 'red'
            
            # Plot eigenvalues
            ax.scatter(eigs.real, eigs.imag, s=30, c=color, alpha=0.6)
            
            # Plot unit circle
            ax.plot(unit_circle_x, unit_circle_y, 'k--', linewidth=1, alpha=0.5)
            
            ax.set_aspect('equal')
            ax.set_xlabel('Real')
            ax.set_ylabel('Imag')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    format_plot(axes)
    plt.show()


# ============================================================================
# MAIN USAGE
# ============================================================================

def find_fixed_points_and_plot_spectra(input_vec, traj_data, params, network, CONFIG, n_attempts=100, tol=1e-7):
    """
    Find fixed points and plot their eigenspectra.
    
    Args:
        input_vec: Input context vector
        traj_data: Dictionary with 'actor_hidden' trajectory data
        params: Network parameters
        network: Network instance
        CONFIG: Config dictionary with 'hidden_size'
        n_attempts: Number of random initializations
    """
    print(f"Searching for fixed points ({n_attempts} attempts)...\n")
    
    # Create batch of random initializations from trajectory
    key = jax.random.PRNGKey(42)
    indices = jax.random.randint(key, (n_attempts,), 0, traj_data['actor_hidden'].shape[0])
    h_inits = traj_data['actor_hidden'][indices, :]
    
    # Find fixed points in batch
    fixed_points, converged, losses = find_fixed_points_batch(
        params=params,
        network=network,
        input_vec=input_vec,
        h_inits=h_inits,
        max_steps=60000,
        learning_rate=0.001,
        tolerance=tol,
        verbose=True
    )
    
    # Filter unique fixed points
    unique_fps = filter_unique_fixed_points(
        fixed_points, converged, uniqueness_threshold=0.1
    )
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found {len(unique_fps)} unique fixed points")
    print(f"{'='*60}\n")
    
    if len(unique_fps) > 0:
        # Analyze stability
        eigenvalues = analyze_stability_batch(unique_fps, input_vec, params, network)

        print(eigenvalues.shape)

        # Plot first 6 eigenspectra
        plot_eigenspectra_grid(eigenvalues, n_plots=6)
        plot_eigenvalue_real_histograms(eigenvalues, bin_size=0.05, n_plots=6)
    else:
        print("No fixed points found.")

    return fixed_points[converged,:], losses[converged]


def run_states_forward(
    h_inits,
    input_vec,
    params,
    network,
    n_steps=100,
    use_self_action=False,
    obs_size=4,
    action_size=2,
):
    """
    Run multiple initial hidden states forward through the RNN.

    Args:
        h_inits: Initial hidden states (batch_size, hidden_size)
        input_vec: Fixed input context (input_size,). When use_self_action=True,
            the obs and reward portions are kept fixed while the action portion
            (input_vec[obs_size:obs_size+action_size]) is replaced each step
            by the actor's own argmax action.
        params: Network parameters
        network: A2CRNNFlax instance
        n_steps: Number of forward steps to simulate
        use_self_action: If True, feed the actor's argmax action back as input.
        obs_size: Length of the observation slice in input_vec.
        action_size: Length of the one-hot action slice in input_vec.

    Returns:
        trajectories: Array of shape (n_steps, batch_size, hidden_size)
    """
    batch_size = h_inits.shape[0]
    hidden_size = h_inits.shape[1]

    trajectories = jnp.zeros((n_steps, batch_size, hidden_size))
    trajectories = trajectories.at[0].set(h_inits)

    h_current = h_inits
    # Per-state input matrix; initially all rows are input_vec
    x_current = jnp.tile(input_vec[None, :], (batch_size, 1))

    for t in range(1, n_steps):
        if use_self_action:
            (h_current, _) = network.apply(
                params, h_current, x_current,
                method=lambda module, h, x: module.rnn_actor(h, x)
            )
            logits = network.apply(
                params, h_current,
                method=lambda module, h: module.actor(h)
            )
            actions = jnp.argmax(logits, axis=-1)           # (batch_size,)
            action_onehot = jax.nn.one_hot(actions, action_size)
            x_current = jnp.concatenate([
                jnp.tile(input_vec[:obs_size][None, :], (batch_size, 1)),
                action_onehot,
                jnp.tile(input_vec[obs_size + action_size:][None, :], (batch_size, 1)),
            ], axis=1)
        else:
            h_current = rnn_step_batch(h_current, input_vec, params, network)

        trajectories = trajectories.at[t].set(h_current)

    return trajectories

def plot_eigenvalue_real_histograms(
    eigenvalues_list,
    bin_size=0.1,
    n_plots=9,
    figsize=(12, 9)
):
    """
    Plot histograms of real parts of eigenvalues as connected scatter plots.
    
    Args:
        eigenvalues_list: Array of eigenvalues (n_points, hidden_size)
        bin_size: Width of histogram bins (default: 0.1)
        n_plots: Number of plots to show (default: 9 for 3x3 grid)
        figsize: Figure size
    """
    n_fps = min(len(eigenvalues_list), n_plots)
    
    if n_fps == 0:
        print("No fixed points to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(9):
        ax = axes[i]
        
        if i < n_fps:
            eigs = eigenvalues_list[i]
            real_parts = eigs.real
            
            # Determine bin edges
            min_real = jnp.floor(jnp.min(real_parts) / bin_size) * bin_size
            max_real = jnp.ceil(jnp.max(real_parts) / bin_size) * bin_size
            bins = jnp.arange(min_real, max_real + bin_size, bin_size)
            
            # Compute histogram
            counts, _ = jnp.histogram(real_parts, bins=bins)
            bin_centers = bins[:-1] + bin_size / 2
            
            # Determine stability
            max_eig = jnp.max(jnp.abs(eigs))
            is_stable = max_eig < 1.0
            color = 'green' if is_stable else 'red'
            
            # Plot as connected scatter
            ax.plot(bin_centers, counts, 'o-', color=color, 
                   linewidth=2, markersize=6, alpha=0.7)
            
            # Add vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='--', 
                      linewidth=1, alpha=0.3)
            
            # Add vertical line at x=1
            ax.axvline(x=1, color='red', linestyle='--', 
                      linewidth=1, alpha=0.3, label='Re=1')
            
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Frequency')
            ax.set_title(f'FP {i+1} ({"Stable" if is_stable else "Unstable"})')
            ax.grid(True, alpha=0.3)
            
        else:
            ax.axis('off')
    
    plt.suptitle('Eigenvalue Real Part Distributions', fontsize=14, y=1.00)
    plt.tight_layout()


# --- (from cell d2c62890-3169-4c28-b90a-6f62b24c079c) ---
def _style_3d_ax(ax, norm, cmap, label, cbar=True):
    """Shared 3-D axis styling: no box, arrow axes, colorbar."""
    ax.set_axis_off()
    ax.autoscale_view()
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ox, oy, oz = xlim[0], ylim[0], zlim[0]
    xl = (xlim[1] - xlim[0]) * 0.45
    yl = (ylim[1] - ylim[0]) * 0.45
    zl = (zlim[1] - zlim[0]) * 0.45
    kw = dict(color='k', arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(ox, oy, oz, xl, 0,  0,  **kw)
    ax.quiver(ox, oy, oz, 0,  yl, 0,  **kw)
    ax.quiver(ox, oy, oz, 0,  0,  zl, **kw)
    ax.text(ox + xl * 1.12, oy,             oz,             'PC 1', ha='center', va='center')
    ax.text(ox,             oy + yl * 1.12, oz,             'PC 2', ha='center', va='center')
    ax.text(ox,             oy,             oz + zl * 1.12, 'PC 3', ha='center', va='center')
    if cbar:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          label=label, shrink=0.35, aspect=15)
        cb.outline.set_visible(False)


def plot_points(points, color_by=None, color_label='', figsize=(6, 5)):
    """Plot an array of hidden states as a 3-D scatter in the first 3 PCs.

    Shows the real-data cloud as a grey background and each point coloured
    by color_by (or all red when color_by is None).

    Args:
        points:       Array (N, hidden_size).
        color_by:     Optional array (N,) used to colour the points.
                      If None, all points are drawn in red.
        color_label:  Colorbar label (ignored when color_by is None).
        figsize:      Figure size tuple.
    """
    points = np.array(points)
    proj   = pca.transform(points)[:, :3]          # (N, 3)

    cmap = plt.cm.coolwarm
    if color_by is not None:
        c_vals = np.array(color_by, dtype=float)
        norm   = Normalize(vmin=c_vals.min(), vmax=c_vals.max())
        colors = cmap(norm(c_vals))
    else:
        c_vals = np.zeros(len(points))
        norm   = Normalize(vmin=0, vmax=1)
        colors = ['red'] * len(points)

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    # Grey background cloud
    downsample_factor = 20
    downsample_vec = np.zeros((pc_activities.shape[0],), dtype=int)
    downsample_vec[:(pc_activities.shape[0] // downsample_factor)] = 1
    np.random.shuffle(downsample_vec)
    ax.scatter(pc_activities[downsample_vec > 0, 0], pc_activities[downsample_vec > 0, 1], pc_activities[downsample_vec > 0, 2],
               s=0.1, color='#d7d9d7', alpha=0.1, rasterized=True, zorder=0)

    # Points of interest
    sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                    s=20, c=c_vals if color_by is not None else 'red',
                    cmap=cmap if color_by is not None else None,
                    norm=norm if color_by is not None else None,
                    alpha=1, zorder=3, depthshade=False)

    _style_3d_ax(ax, norm, cmap, color_label if color_by is not None else '')
    plt.tight_layout()
    plt.show()


# --- (from cell 714874fb-bebe-4f66-96a6-4887c1413b15) ---
def plot_trajectory_dyn_3d(traces, pca, color_traces=None, start=0, end=None, figsize=(7, 6), cbar_label=None, params=None, elev=20, azim=30):
    """3-D version of plot_trajectory_dyn: plots the first 3 PCs of traces,
    coloured by color_traces (or time if None), against the background cloud.

    Parameters
    ----------
    traces : ndarray, shape (T, batch, hidden_size)
    color_traces : ndarray, shape (T, batch) or (T,), optional
        Values used to colour each segment. If None, colours by time step.
    start, end : int  – time-step slice to plot
    cbar_label : str, optional
    """
    if end is None:
        end = traces.shape[0]
    t_slice = slice(start, end)

    cmap = plt.cm.viridis

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Background cloud of all activity
    # ax.scatter(
    #     pc_activities[:, 0], pc_activities[:, 1], pc_activities[:, 2],
    #     s=0.1, color='#d7d9d7', alpha=0.1, rasterized=True, zorder=-3,
    # )

    # Determine colour values and normalisation
    if color_traces is None:
        color_vals_all = [np.arange(end - start)] * traces.shape[1]
        norm = Normalize(vmin=0, vmax=end - start)
        label = cbar_label or 'time step'
    else:
        ct = np.array(color_traces)
        # Support (T, batch) or (T,) / (T, 1)
        if ct.ndim == 1:
            ct = ct[:, None]
        ct = ct[t_slice]
        color_vals_all = [ct[:, i] if ct.shape[1] > 1 else ct[:, 0] for i in range(traces.shape[1])]
        all_vals = np.concatenate(color_vals_all)
        norm = Normalize(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
        label = cbar_label or 'value'

    # Trajectories
    for i in range(traces.shape[1]):
        proj = pca.transform(traces[t_slice, i, :])[:, :3]
        color_vals = color_vals_all[i]

        pts = proj.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.9)
        lc.set_array(np.asarray(color_vals[:-1]).ravel())
        ax.add_collection3d(lc)

    # Turn off the default axis box and draw clean arrows from a common origin
    ax.set_axis_off()
    ax.autoscale_view()
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ox, oy, oz = xlim[0], ylim[0], zlim[0]
    xl = (xlim[1] - xlim[0]) * 0.45
    yl = (ylim[1] - ylim[0]) * 0.45
    zl = (zlim[1] - zlim[0]) * 0.45
    kw = dict(color='k', arrow_length_ratio=0.08, linewidth=1)
    ax.quiver(ox, oy, oz, xl, 0,  0,  **kw)
    ax.quiver(ox, oy, oz, 0,  yl, 0,  **kw)
    ax.quiver(ox, oy, oz, 0,  0,  zl, **kw)
    ax.text(ox + xl * 1.12, oy,            oz,            'PC 1', ha='center', va='center')
    ax.text(ox,            oy + yl * 1.12, oz,            'PC 2', ha='center', va='center')
    ax.text(ox,            oy,            oz + zl * 1.12, 'PC 3', ha='center', va='center')
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label, shrink=0.35, aspect=15)
    cbar.outline.set_visible(False)
    if params is not None:
        add_separatrix_plane(ax, params, pca)
    plt.tight_layout()
    plt.show()


# --- (from cell d4b242e1-af53-4c7b-8673-52409b4a5330) ---
def extract_patch_starts_and_stops(agent_in_patch):
    """
    Find contiguous periods where the agent is in a patch.

    Parameters
    ----------
    agent_in_patch : ndarray, shape (T, n_trials), bool

    Returns
    -------
    list of n_trials arrays, each shape (n_periods, 2)
        Each row is [start, end) index along the time axis.
    """
    T, n_trials = agent_in_patch.shape[:2]
    results = []

    for trial in range(n_trials):
        signal = agent_in_patch[:, trial].astype(bool)

        padded = np.concatenate([[False], signal, [False]])
        diff = np.diff(padded.astype(np.int8))

        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        results.append(np.stack([starts, ends], axis=1))

    return results


# --- (from cell add-separatrix-plane) ---
def add_separatrix_plane(ax, params, pca, n_pts=500, alpha=0.2, color='steelblue'):
    """Overlay the action decision-boundary plane onto a 3D PC-space axis.

    The separatrix is the hyperplane in hidden space where the two action logits
    are equal: (kernel[:,0] - kernel[:,1]) · h + (bias[0] - bias[1]) = 0.
    It is projected into the first 3 PCs and drawn as a surface.
    """
    kernel = np.array(params['params']['actor']['kernel'])  # (hidden, 2)
    bias   = np.array(params['params']['actor']['bias'])    # (2,)

    # Decision-boundary normal in hidden space: w·h + b_diff = 0
    w      = kernel[:, 0] - kernel[:, 1]   # (hidden,)
    b_diff = float(bias[0] - bias[1])

    # Project to 3D PC space: n_pc · z = d
    n_pc = w @ pca.components_[:3].T          # (3,)
    d    = -(np.dot(w, pca.mean_) + b_diff)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, n_pts), np.linspace(*ylim, n_pts))

    if np.abs(n_pc[2]) < 1e-8:
        return  # plane nearly vertical in z — skip

    zz = (d - n_pc[0] * xx - n_pc[1] * yy) / n_pc[2]
    
    def filter_z(z, max):
        z = np.where(np.abs(z) > max, np.nan, z)
        return z

    zz = filter_z(zz, np.abs(np.array(ax.get_zlim())).max())

    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color, shade=False, zorder=0)


# --- (from cell 603947a4-78ec-4e47-8215-9cd37f44d61f) ---
def plot_patch_trajectories_3d(traj_data, threshold=0.5, figsize=(5, 4), max_trajectories=None,
                                color_by='time', params=None, pca=None, elev=20, azim=30):
    """
    Three 3-D PC plots of hidden-state trajectories during patches, shown side by side.

    A patch is a contiguous run where traj_data['observations'][..., 0] > threshold.

    Plot 1 – patches where obs[..., 1] exceeds threshold at any point in the patch.
    Plot 2 – patches where obs[..., 2] exceeds threshold at any point in the patch.
    Plot 3 – patches where obs[..., 3] exceeds threshold at any point in the patch.

    color_by : 'time' (default) or a key in traj_data whose values have the same
               leading (T, n_trials) shape as observations, e.g. 'rewards_seen_in_patch'.
               If the array has a trailing feature dimension its first column is used.
    params   : if provided, the action separatrix plane is overlaid on each panel.
    elev     : camera elevation angle in degrees (default 20).
    azim     : camera azimuth angle in degrees (default 30).
    """
    _steps_per_env = 20000
    _n_envs = max(1, traj_data['actor_hidden'].shape[0] // _steps_per_env)
    obs     = traj_data['observations'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)   # (T, n_trials, obs_dim)
    hidden  = traj_data['actor_hidden'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)   # (T, n_trials, hidden_size)
    in_patch = traj_data['agent_in_patch'].reshape(_n_envs, _steps_per_env).T                     # (T, n_trials)
    periods = extract_patch_starts_and_stops(in_patch)

    # Pre-compute color signal array if not time-based
    if color_by != 'time':
        raw = traj_data[color_by].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)  # (T, n_trials, ...)
        color_signal = raw[..., 0]   # (T, n_trials), take first feature column
        cbar_label = color_by
    else:
        color_signal = None
        cbar_label = 'time in patch'

    _base_cmap = copy(plt.cm.viridis)
    dims = [(1, 'obs dim 1 high'), (2, 'obs dim 2 high'), (3, 'obs dim 3 high')]

    # Pre-collect all projections to compute global axis limits
    all_projs_global = []
    for trial_periods in periods:
        for start, end in trial_periods:
            proj = pca.transform(hidden[start:end, :, :].reshape(-1, hidden.shape[-1]))[:, :3]
            all_projs_global.append(proj)
    if all_projs_global:
        all_pts = np.concatenate(all_projs_global, axis=0)
        global_lims = [(all_pts[:, i].min(), all_pts[:, i].max()) for i in range(3)]
    else:
        global_lims = None

    fig = plt.figure(figsize=(figsize[0], figsize[1] * 3))
    axes = [fig.add_subplot(3, 1, k + 1, projection='3d') for k in range(3)]

    for ax, (dim, title) in zip(axes, dims):
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

        # Collect all patch projections + colour values so we can share norm
        patch_projs  = []
        patch_colors = []
        for trial, trial_periods in enumerate(periods):
            for start, end in trial_periods:
                patch_obs = obs[start:end, trial, dim]
                if not np.any(patch_obs > threshold):
                    continue
                proj = pca.transform(hidden[start:end, trial, :])[:, :3]
                patch_projs.append(proj)
                if color_signal is not None:
                    patch_colors.append(color_signal[start:end, trial])
                else:
                    patch_colors.append(np.arange(len(proj)))

        if not patch_projs:
            print(f'No patches found for dim {dim}')
            ax.set_visible(False)
            continue

        if max_trajectories is not None:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(patch_projs), size=min(max_trajectories, len(patch_projs)), replace=False)
            patch_projs  = [patch_projs[i]  for i in idx]
            patch_colors = [patch_colors[i] for i in idx]

        all_vals = np.concatenate([c.ravel() for c in patch_colors])
        if color_by == 'reward_site_idx':
            cmap = copy(_base_cmap)
            cmap.set_under('black')
            norm = Normalize(vmin=0, vmax=np.nanmax(all_vals))
        else:
            cmap = _base_cmap
            norm = Normalize(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))

        for proj, cvals in zip(patch_projs, patch_colors):
            pts  = proj.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=1.0, alpha=0.6)
            lc.set_array(np.asarray(cvals[:-1]).ravel())
            ax.add_collection3d(lc)

        _style_3d_ax(ax, norm, cmap, cbar_label)

        if global_lims is not None:
            ax.set_xlim(*global_lims[0])
            ax.set_ylim(*global_lims[1])
            ax.set_zlim(*global_lims[2])

        if params is not None:
            add_separatrix_plane(ax, params, pca)

    plt.tight_layout()
    return fig


def animate_patch_trajectories_3d(
    traj_data,
    pca,
    save_path,
    threshold=0.5,
    max_trajectories=None,
    color_by='time',
    params=None,
    elev1=20,
    azim1=30,
    elev2=20,
    azim2=120,
    fps=30,
    steps_per_frame=1,
    figsize=(10, 12),
    rotation=False,
    rotation_seconds=3,
):
    """Animate patch trajectories in 3D PC space, building each trajectory step by step.

    Saves a video to save_path (.mp4 preferred, .gif also works).

    rotation=False (default): 3 rows × 2 columns; both viewpoints (elev1/azim1 and
        elev2/azim2) are shown side by side simultaneously throughout the video.

    rotation=True: 3 rows × 1 column.  The video plays in three phases:
        1. Trajectory builds up from view 1 (elev1/azim1).
        2. Rotation: the view smoothly sweeps from view 1 to view 2 over
           rotation_seconds seconds, with the full trajectory visible.
        3. Trajectory builds up again from view 2 (elev2/azim2).

    Ghost lines show the full trajectory faintly; a growing colored segment
    is drawn on top as the animation progresses.

    steps_per_frame : how many timesteps to advance per video frame (increase
                      to speed up long trajectories).
    rotation_seconds: duration of the rotation sweep between the two views.
    """
    _steps_per_env = 20000
    _n_envs = max(1, traj_data['actor_hidden'].shape[0] // _steps_per_env)
    obs    = traj_data['observations'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    hidden = traj_data['actor_hidden'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(_n_envs, _steps_per_env).T
    periods = extract_patch_starts_and_stops(in_patch)

    if color_by != 'time':
        raw = traj_data[color_by].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
        color_signal = raw[..., 0]
        cbar_label = color_by
    else:
        color_signal = None
        cbar_label = 'time in patch'

    _base_cmap = copy(plt.cm.viridis)
    dims = [(1, 'obs dim 1 high'), (2, 'obs dim 2 high'), (3, 'obs dim 3 high')]

    # --- collect per-dim patch projections and colour values ---
    per_dim_data = []
    all_projs_global = []

    for dim, _ in dims:
        patch_projs, patch_colors = [], []
        for trial, trial_periods in enumerate(periods):
            for start, end in trial_periods:
                patch_obs = obs[start:end, trial, dim]
                if not np.any(patch_obs > threshold):
                    continue
                proj = pca.transform(hidden[start:end, trial, :])[:, :3]
                patch_projs.append(proj)
                patch_colors.append(
                    color_signal[start:end, trial] if color_signal is not None
                    else np.arange(len(proj), dtype=float)
                )
                all_projs_global.append(proj)

        if max_trajectories is not None and patch_projs:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(patch_projs), size=min(max_trajectories, len(patch_projs)), replace=False)
            patch_projs  = [patch_projs[i]  for i in idx]
            patch_colors = [patch_colors[i] for i in idx]

        per_dim_data.append((patch_projs, patch_colors))

    # global axis limits
    if all_projs_global:
        all_pts = np.concatenate(all_projs_global, axis=0)
        global_lims = [(all_pts[:, i].min(), all_pts[:, i].max()) for i in range(3)]
    else:
        global_lims = None

    max_len = max(
        (len(p) for projs, _ in per_dim_data for p in projs),
        default=1,
    )
    n_frames = int(np.ceil(max_len / steps_per_frame))

    # compute per-dim colour norms once (shared by both rotation and non-rotation paths)
    norms = []
    cmaps = []
    for patch_projs, patch_colors in per_dim_data:
        if patch_colors:
            all_vals = np.concatenate([c.ravel() for c in patch_colors])
            if color_by == 'reward_site_idx':
                cmap = copy(_base_cmap)
                cmap.set_under('black')
                norm = Normalize(vmin=0, vmax=np.nanmax(all_vals))
            else:
                cmap = _base_cmap
                norm = Normalize(vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
        else:
            norm, cmap = Normalize(0, 1), _base_cmap
        norms.append(norm)
        cmaps.append(cmap)

    def _draw_panel(ax, patch_projs, patch_colors, norm, cmap, elev, azim, title):
        ax.cla()
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=elev, azim=azim)

        if global_lims is not None:
            ax.set_xlim(*global_lims[0])
            ax.set_ylim(*global_lims[1])
            ax.set_zlim(*global_lims[2])

        for proj, cvals in zip(patch_projs, patch_colors):
            ax.plot(proj[:, 0], proj[:, 1], proj[:, 2],
                    color='grey', alpha=0.15, linewidth=0.5)

            end_t = min(t, len(proj))
            if end_t < 2:
                continue
            seg_proj  = proj[:end_t]
            seg_cvals = np.asarray(cvals[:end_t - 1]).ravel()
            pts  = seg_proj.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(segs, cmap=cmap, norm=norm,
                                    linewidth=1.2, alpha=0.8)
            lc.set_array(seg_cvals)
            ax.add_collection3d(lc)

        if params is not None:
            add_separatrix_plane(ax, params, pca)

        _style_3d_ax(ax, norm, cmap, cbar_label, cbar=False)

    t = 0

    if rotation:
        # --- build figure: 3 rows × 1 column ---
        fig = plt.figure(figsize=(figsize[0] // 2, figsize[1]))
        row_axes = [fig.add_subplot(3, 1, row + 1, projection='3d') for row in range(3)]

        for ax, norm, cmap in zip(row_axes, norms, cmaps):
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                         label=cbar_label, shrink=0.35, aspect=15).outline.set_visible(False)

        n_rot_frames = int(fps * rotation_seconds)
        total_frames = 2 * n_frames + n_rot_frames

        def _draw_frame(frame):
            nonlocal t
            if frame < n_frames:
                # Phase 1: build trajectory at view 1
                t = int((frame + 1) * steps_per_frame)
                elev, azim = elev1, azim1
            elif frame < n_frames + n_rot_frames:
                # Phase 2: rotate with full trajectory visible
                t = max_len
                alpha = (frame - n_frames) / n_rot_frames
                elev = elev1 + alpha * (elev2 - elev1)
                azim = azim1 + alpha * (azim2 - azim1)
            else:
                # Phase 3: build trajectory at view 2
                t = int((frame - n_frames - n_rot_frames + 1) * steps_per_frame)
                elev, azim = elev2, azim2

            for ax, (dim, title), (patch_projs, patch_colors), norm, cmap in zip(
                row_axes, dims, per_dim_data, norms, cmaps
            ):
                _draw_panel(ax, patch_projs, patch_colors, norm, cmap, elev, azim, title)

            fig.suptitle(f'step {t}', fontsize=10)

        ani = animation.FuncAnimation(
            fig, _draw_frame, frames=total_frames, interval=1000 / fps
        )

    else:
        # --- build figure: 3 rows × 2 columns (both views simultaneously) ---
        fig = plt.figure(figsize=figsize)
        axes = [
            [fig.add_subplot(3, 2, row * 2 + col + 1, projection='3d') for col in range(2)]
            for row in range(3)
        ]
        view_params = [(elev1, azim1), (elev2, azim2)]

        # Attach each colorbar to both axes in its row so matplotlib positions it sensibly.
        for pair, norm, cmap in zip(axes, norms, cmaps):
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=pair,
                         label=cbar_label, shrink=0.35, aspect=15).outline.set_visible(False)

        def _draw_frame(frame):
            nonlocal t
            t = int((frame + 1) * steps_per_frame)

            for row_axes, (dim, title), (patch_projs, patch_colors), norm, cmap in zip(
                axes, dims, per_dim_data, norms, cmaps
            ):
                for ax, (elev, azim) in zip(row_axes, view_params):
                    _draw_panel(ax, patch_projs, patch_colors, norm, cmap, elev, azim, title)

            fig.suptitle(f'step {t}', fontsize=10)

        ani = animation.FuncAnimation(
            fig, _draw_frame, frames=n_frames, interval=1000 / fps
        )

    save_path = str(save_path)
    if save_path.endswith('.gif') or not animation.FFMpegWriter.isAvailable():
        if not save_path.endswith('.gif'):
            save_path = save_path.rsplit('.', 1)[0] + '.gif'
            print(f'ffmpeg not found — saving as GIF instead: {save_path}')
        ani.save(save_path, writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(save_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))

    plt.close(fig)
    print(f'Saved to {save_path}')


# --- (from cell 1f017d61-4055-4a0c-9f44-18729ea9bc9a) ---
def plot_patch_activity_heatmaps(traj_data, threshold=0.5, n_examples=3, figsize=(4, 3)):
    """
    For each of the 3 obs dims (1, 2, 3), plot n_examples patch trajectories as
    heatmaps of raw hidden-unit activity (rows = units, columns = time steps).
    Patches are selected randomly from those where the given dim exceeds threshold.
    """
    _steps_per_env = 20000
    _n_envs = max(1, traj_data['actor_hidden'].shape[0] // _steps_per_env)
    obs    = traj_data['observations'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    hidden = traj_data['actor_hidden'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(_n_envs, _steps_per_env).T
    periods = extract_patch_starts_and_stops(in_patch)

    dims = [(1, 'obs dim 1 high'), (2, 'obs dim 2 high'), (3, 'obs dim 3 high')]

    rng = np.random.default_rng(0)

    for dim, dim_title in dims:
        # Collect qualifying patches
        patches = []
        for trial, trial_periods in enumerate(periods):
            for start, end in trial_periods:
                if np.any(obs[start:end, trial, dim] > threshold):
                    patches.append(hidden[start:end, trial, :])  # (T_patch, hidden)

        if not patches:
            print(f'No patches found for {dim_title}')
            continue

        chosen = rng.choice(len(patches), size=min(n_examples, len(patches)), replace=False)

        fig, axes = plt.subplots(
            1, len(chosen),
            figsize=(figsize[0] * len(chosen), figsize[1]),
            sharey=True,
        )
        if len(chosen) == 1:
            axes = [axes]

        fig.suptitle(dim_title)

        # Shared colour scale across examples
        vmin = min(patches[i].min() for i in chosen)
        vmax = max(patches[i].max() for i in chosen)

        for ax, idx in zip(axes, chosen):
            act = patches[idx].T   # (units, time)
            im = ax.imshow(
                act,
                aspect='auto',
                interpolation='nearest',
                cmap='RdBu_r',
                vmin=vmin, vmax=vmax,
                origin='upper',
            )
            ax.set_xlabel('Time step')
            if ax is axes[0]:
                ax.set_ylabel('Unit')
            else:
                ax.set_yticks([])

        cbar = fig.colorbar(im, ax=axes[-1], shrink=0.8)
        cbar.outline.set_visible(False)
        cbar.set_label('Activity')
        plt.tight_layout()
        plt.show()


# --- (from cell participation-ratio) ---
def participation_ratio(X):
    """
    Estimate the dimensionality of a point cloud using the participation ratio (D_PR).

    D_PR = (sum_i lambda_i)^2 / (sum_i lambda_i^2)

    where lambda_i are the eigenvalues of the covariance matrix of X (equivalent
    to the squared singular values / variance explained by each PC).

    Reference: Recanatesi et al., Patterns (2021), eq. for D_PR.

    Parameters
    ----------
    X : (N, d) array — N observations of d-dimensional activity.

    Returns
    -------
    float — participation ratio, in range [1, d].
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    lambdas = s ** 2          # proportional to eigenvalues of the covariance matrix
    return float(lambdas.sum() ** 2 / (lambdas ** 2).sum())



# --- pre-odor onset detection ---
def find_pre_odor_onset_states(traj_data, max_lookahead=3, patch_num=None):
    """
    Detect reward-site entries via np.diff(reward_site_idx) > 0 and return
    hidden states and metadata for each qualifying event.

    Sites where the incoming reward_site_idx is zero are excluded.

    Timing
    ------
    diff[t] > 0  →  reward_site_idx increases from step t to step t+1
      pre_t   = t      (last step before the new site)
      onset_t = t + 1  (first step at the new site)

    Returns
    -------
    dict with N-length arrays (N = number of qualifying events):
      'hidden'      : (N, max_lookahead+1, H)  states at pre_t, onset_t, …, onset_t+max_lookahead-1
      'reward_site' : (N,)   reward_site_idx at onset_t
      'isi'         : (N,)   inter_odor_site_distances at onset_t
      'in_patch'    : (N,)   bool, agent_in_patch at pre_t
      'trial_idx'   : (N,)   trial index
      'pre_t_idx'   : (N,)   pre_t timestep index within trial
    """
    _steps_per_env = 20000
    _n_envs = max(1, traj_data['actor_hidden'].shape[0] // _steps_per_env)
    hidden   = traj_data['actor_hidden'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    rsi      = traj_data['reward_site_idx'].reshape(_n_envs, _steps_per_env).T
    isi_raw  = traj_data['inter_odor_site_distances'].reshape(_n_envs, _steps_per_env, -1)[:, :, 0].T
    patch_nums = traj_data['current_patch_num'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)[:, :, 0]
    T, n_trials, _ = hidden.shape

    h_traj, rsi_list, isi_list = [], [], []
    trial_list, pret_list = [], []

    for trial in range(n_trials):
        for t in np.where(np.diff(rsi[:, trial]) > 0)[0]:
            onset_t = t + 2
            if onset_t - 1 + max_lookahead >= T:
                continue
            if rsi[onset_t, trial] == 0:
                continue
            if patch_num is not None and patch_nums[onset_t, trial] != patch_num:
                continue
            h_traj.append(hidden[onset_t - 1:onset_t + max_lookahead, trial])
            rsi_list.append(float(rsi[onset_t, trial]))
            isi_list.append(float(isi_raw[onset_t, trial]))
            trial_list.append(trial)
            pret_list.append(onset_t - 1)

    return {
        'hidden':      np.array(h_traj),
        'reward_site': np.array(rsi_list),
        'isi':         np.array(isi_list),
        'trial_idx':   np.array(trial_list, dtype=int),
        'pre_t_idx':   np.array(pret_list,  dtype=int),
    }


def fit_remaining_time_decoder(traj_data):
    """
    Fit a linear decoder from pre-odor hidden states to remaining time in patch.

    For each pre-odor onset event, remaining time is the number of timesteps from
    that state until the agent is no longer in the patch (obs[:,0] == 0).

    Returns:
        decoder_dir: (H,) unit vector oriented so positive projection = less time remaining.
        Returns None if too few valid events are found.
    """
    events    = find_pre_odor_onset_states(traj_data)
    if len(events['hidden']) == 0:
        return None

    _steps_per_env = 20000
    _n_envs = max(1, traj_data['observations'].shape[0] // _steps_per_env)
    obs_all   = traj_data['observations'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    in_patch  = obs_all[:, :, 0]  # (T, n_trials)
    T         = in_patch.shape[0]

    remaining_times = []
    valid_mask      = []
    for i in range(len(events['hidden'])):
        trial = events['trial_idx'][i]
        pre_t = events['pre_t_idx'][i]
        future = in_patch[pre_t:, trial]
        exits  = np.where(future == 0)[0]
        if len(exits) > 0:
            remaining_times.append(int(exits[0]))
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    if valid_mask.sum() < 10:
        return None

    h_valid = events['hidden'][valid_mask, 0]
    y       = np.array(remaining_times, dtype=float)

    reg = LinearRegression(fit_intercept=True).fit(h_valid, y)
    w   = reg.coef_.copy()

    # Orient: positive projection = less time remaining
    if np.corrcoef(h_valid @ w, y)[0, 1] > 0:
        w = -w

    return w / np.linalg.norm(w)


def collect_pre_odor_time_to_exit(traj_data, patch_num=None):
    """
    For every pre-odor onset state detected by find_pre_odor_onset_states, scan
    forward in the trajectory to find how many steps remain until the agent leaves
    the patch (in_patch == 0).  Events where no exit is observed are discarded.

    Returns a dict:
      hidden        (N, H) float — pre-odor hidden states
      time_to_exit  (N,)   float — steps from that state until patch exit
    """
    events   = find_pre_odor_onset_states(traj_data, patch_num=patch_num)
    if len(events['hidden']) == 0:
        return dict(hidden=np.empty((0, 1)), time_to_exit=np.empty(0))

    _steps_per_env = 20000
    _n_envs = max(1, traj_data['observations'].shape[0] // _steps_per_env)
    obs_all  = traj_data['observations'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    in_patch = obs_all[:, :, 0]   # (T, n_trials)

    times = []
    valid = []
    for i in range(len(events['hidden'])):
        trial  = events['trial_idx'][i]
        pre_t  = events['pre_t_idx'][i]
        future = in_patch[pre_t:, trial]
        exits  = np.where(future == 0)[0]
        if len(exits) > 0:
            times.append(int(exits[0]))
            valid.append(True)
        else:
            valid.append(False)

    valid = np.array(valid)
    return dict(
        hidden       = events['hidden'][valid, 0],
        time_to_exit = np.array(times, dtype=float),
    )


def knn_time_to_exit(h_query, tte_data, n_neighbors=10, chunk_size=512):
    """
    Predict time to patch exit for one or more hidden states using K-nearest
    neighbours over pre-odor states collected by collect_pre_odor_time_to_exit.

    Distances are computed via ||a-b||² = ||a||² + ||b||² - 2aᵀb in chunks of
    chunk_size queries, keeping peak memory to O(chunk_size × N) rather than
    O(M × N × H).

    Parameters
    ----------
    h_query     : (H,) or (M, H) — query hidden state(s)
    tte_data    : dict returned by collect_pre_odor_time_to_exit
    n_neighbors : number of neighbours to average
    chunk_size  : number of query rows processed at once

    Returns
    -------
    (M,) predicted times, or a scalar if h_query was 1-D.
    """
    h_ref   = tte_data['hidden']        # (N, H)
    tte     = tte_data['time_to_exit']  # (N,)
    scalar  = h_query.ndim == 1
    h_query = np.atleast_2d(np.array(h_query, dtype=np.float32))  # (M, H)
    h_ref   = np.array(h_ref, dtype=np.float32)

    M           = len(h_query)
    k           = min(n_neighbors, len(h_ref))
    predictions = np.empty(M, dtype=np.float64)
    ref_sq      = (h_ref ** 2).sum(axis=1)   # (N,) — precomputed once

    for start in range(0, M, chunk_size):
        end   = min(start + chunk_size, M)
        chunk = h_query[start:end]                                 # (C, H)
        # squared L2: ||chunk_i - ref_j||² = ||chunk_i||² + ||ref_j||² - 2 chunk_i·ref_j
        sq_dists = (chunk ** 2).sum(axis=1, keepdims=True) + ref_sq - 2 * (chunk @ h_ref.T)
        np.maximum(sq_dists, 0, out=sq_dists)                      # numerical safety
        nn_idx = np.argpartition(sq_dists, k - 1, axis=-1)[:, :k]
        for i in range(len(chunk)):
            predictions[start + i] = tte[nn_idx[i]].mean()

    return float(predictions[0]) if scalar else predictions


def patch_progress(h_query, tte_data, n_neighbors=10, chunk_size=512):
    """Patch progress = 1 - knn_time_to_exit / p98_time_to_exit, clipped to [0, 1]."""
    p98_tte = float(np.percentile(tte_data['time_to_exit'], 98))
    tte     = knn_time_to_exit(h_query, tte_data, n_neighbors=n_neighbors, chunk_size=chunk_size)
    if np.ndim(tte) == 0:
        return float(np.clip(1.0 - tte / p98_tte, 0.0, 1.0))
    return np.clip(1.0 - np.asarray(tte) / p98_tte, 0.0, 1.0)


# --- (from cell pre-patch-states-3d) ---
def plot_pre_patch_states_3d(traj_data, pca, params=None, figsize=(5, 4), color_by='time',
                              show_background=False,
                              elev1=20, azim1=30, elev2=20, azim2=120):
    """
    Scatter hidden states just before each reward-site entry (in_patch events),
    using find_pre_odor_onset_states for detection.

    Shows a 1×2 figure: two perspectives of the global PCA projection.
    """
    events     = find_pre_odor_onset_states(traj_data)
    sub_hidden = events['hidden'][:, 0]

    if color_by == 'inter_odor_site_distances':
        sub_colors = events['isi']
        cbar_label = 'inter_odor_site_distances'
    elif color_by == 'reward_site_idx':
        sub_colors = events['reward_site']
        cbar_label = 'reward_site_idx'
    elif color_by != 'time':
        _spe = 20000
        _ne  = max(1, traj_data[color_by].shape[0] // _spe)
        raw        = traj_data[color_by].reshape(_ne, _spe, -1)[:, :, 0].T
        sub_colors = raw[events['pre_t_idx'], events['trial_idx']]
        cbar_label = color_by
    else:
        sub_colors = events['pre_t_idx'].astype(float)
        cbar_label = 'time step'

    cbar_label = cbar_label.replace('_', ' ')
    cbar_label = cbar_label[0].upper() + cbar_label[1:]

    if len(sub_hidden) == 0:
        print('No pre-odor onset events found')
        return

    # Patch number for each event
    _spe2 = 20000
    _ne2  = max(1, traj_data['current_patch_num'].shape[0] // _spe2)
    raw_patch   = traj_data['current_patch_num'].reshape(_ne2, _spe2, -1).transpose(1, 0, 2)[:, :, 0]
    event_patch = raw_patch[events['pre_t_idx'], events['trial_idx']].astype(int)

    d_pr_all = participation_ratio(traj_data['actor_hidden'])
    cmap = plt.cm.viridis
    norm = Normalize(vmin=np.nanmin(sub_colors), vmax=np.nanmax(sub_colors))

    # Background: subsample of all states
    all_hidden = traj_data['actor_hidden']
    rng_bg    = np.random.default_rng(1)
    bg_hidden = all_hidden[rng_bg.choice(len(all_hidden), size=min(2000, len(all_hidden)), replace=False)]
    bg_global = pca.transform(bg_hidden)[:, :3]

    fig = plt.figure(figsize=(figsize[0] * 2, figsize[1] * 2))

    def _draw(ax, pts, colors, subtitle, background=True):
        ax.set_title(subtitle, fontsize=8)
        if background:
            ax.scatter(bg_global[:, 0], bg_global[:, 1], bg_global[:, 2],
                       facecolors='none', edgecolors='lightgray', s=6, alpha=0.3,
                       depthshade=True, zorder=0, linewidths=0.5)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors, cmap=cmap, norm=norm, s=4, alpha=0.6, depthshade=True)
        _style_3d_ax(ax, norm, cmap, cbar_label)
        if params is not None:
            add_separatrix_plane(ax, params, pca)
        ax.view_init(elev=elev1, azim=azim1)

    ref_ax = None
    for row, pnum in enumerate([1, 2]):
        mask   = event_patch == pnum
        pts    = pca.transform(sub_hidden[mask])[:, :3]
        colors = sub_colors[mask]
        d_pr_sub = participation_ratio(sub_hidden[mask]) if mask.sum() > 1 else float('nan')
        title_base = f'Patch {pnum} — pre-site states\n$D_{{PR}}$ all={d_pr_all:.1f}  sub={d_pr_sub:.1f}'

        ax_left = fig.add_subplot(2, 2, row * 2 + 1, projection='3d')
        _draw(ax_left, pts, colors, title_base + '\n(with background)')

        ax_right = fig.add_subplot(2, 2, row * 2 + 2, projection='3d')
        _draw(ax_right, pts, colors, title_base + '\n(no background)', background=False)

        # Lock right panel bounds to left
        ax_right.set_xlim(ax_left.get_xlim())
        ax_right.set_ylim(ax_left.get_ylim())
        ax_right.set_zlim(ax_left.get_zlim())

        if ref_ax is None:
            ref_ax = ax_left

    plt.tight_layout()
    return fig


# --- (from cell pre-patch-jac-3d) ---
def plot_pre_patch_states_jac_3d(traj_data, params, network, pca,
                                  figsize=(5, 4), elev=20, azim=30, downsample=4):
    """
    Colors pre-site-entry states by quantities derived from the input Jacobian
    J = dh_next/du.  States are detected via find_pre_odor_onset_states.

    Two rows, one per u_diff context (obs[2] / obs[3] channel).
    Two cols: ||J @ u_diff|| and (J @ u_diff) · w_sep.
    """

    # Separatrix normal (hidden space)
    kernel = np.array(params['params']['actor']['kernel'])  # (H, 2)
    w      = kernel[:, 0] - kernel[:, 1]                      # (H,)
    w_hat  = w / np.linalg.norm(w)

    # Input Jacobian: dh_next/du at (h_prev, u), shape (H, input_dim)
    def _jac_fn(h_prev, u_vec):
        def f(u):
            return network.apply(
                params, h_prev[None], u[None],
                method=lambda m, h, x: m.rnn_actor(h, x)
            )[0][0]
        return jax.jacfwd(f)(u_vec)

    _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))  # compiled per context

    contexts = [
        ('obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        ('obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig = plt.figure(figsize=(figsize[0] * 2, figsize[1] * 2))

    events     = find_pre_odor_onset_states(traj_data)
    sub_hidden = events['hidden'][:, 0]

    for row, (ctx_label, u_ctx, u_diff) in enumerate(contexts):
        if len(sub_hidden) == 0:
            print(f'No pre-odor onset events found for {ctx_label}')
            continue

        # Downsample
        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // downsample), replace=False)
        sub_hidden = sub_hidden[keep]

        # Compute input Jacobians and Jacobian-vector products
        J_batch = np.array(_jac_batch(jnp.array(sub_hidden)))  # (N, H, input_dim)
        Ju      = J_batch @ np.array(u_diff)                    # (N, H)

        scalar_norm = np.linalg.norm(Ju, axis=1)                    # (N,)
        scalar_sep  = Ju @ w_hat                                     # (N,)

        pts = pca.transform(sub_hidden)[:, :3]

        for col, (scalar, cmap_name, cbar_label) in enumerate([
            (scalar_norm, 'viridis',  r'$\| J \cdot u\| $'),
            (scalar_sep,  'coolwarm', r'$(Ju)^T n_{\rm sep}$'),
        ]):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection='3d')
            ax.set_title(f'{ctx_label}\n{cbar_label}', fontsize=8)

            cmap = plt.get_cmap(cmap_name)
            if cmap_name == 'coolwarm':
                abs_max = max(float(np.abs(scalar).max()), 1e-8)
                norm = Normalize(vmin=-abs_max, vmax=abs_max)
            else:
                norm = Normalize(vmin=scalar.min(), vmax=scalar.max())

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=scalar, cmap=cmap, norm=norm,
                       s=10, alpha=0.6, depthshade=True)
            _style_3d_ax(ax, norm, cmap, cbar_label)
            add_separatrix_plane(ax, params, pca)
            ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


# --- (from cell pre-patch-jac-2d) ---
def plot_pre_patch_states_jac_2d(traj_data, params, network,
                                  figsize=(3, 2.2), downsample=4,
                                  n_fp_attempts=200, fp_tol=1e-3,
                                  color_by='reward_site_idx', color_label=None):
    """
    2-D companion to plot_pre_patch_states_jac_3d.  States detected via
    find_pre_odor_onset_states; two rows for the two u_diff contexts.

    X-axis: projection onto FP PC1.
    Y-axis: Jacobian scalar (3 cols: norm, sep projection, sep distance).
    """
    _color_label = color_label or color_by

    # Separatrix normal
    kernel = np.array(params['params']['actor']['kernel'])  # (H, 2)
    bias   = np.array(params['params']['actor']['bias'])    # (2,)
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(bias[0] - bias[1])
    w_norm = np.linalg.norm(w)
    w_hat  = w / w_norm

    # Fixed-point first PC (u_ctx is the same for both contexts)
    u_ctx_fp = jnp.array([1., 0., 0., 0., 0., 1., 0.])
    key      = jax.random.PRNGKey(42)
    indices  = jax.random.randint(key, (n_fp_attempts,), 0,
                                   traj_data['actor_hidden'].shape[0])
    h_inits  = jnp.array(traj_data['actor_hidden'][indices])
    fps, converged, _ = find_fixed_points_batch(
        params=params, network=network, input_vec=u_ctx_fp,
        h_inits=h_inits, max_steps=60000, learning_rate=0.001,
        tolerance=fp_tol, verbose=False,
    )
    unique_fps = np.array(filter_unique_fixed_points(fps, converged))
    assert len(unique_fps) > 0, "No fixed points found"
    fp_pca = PCA(n_components=min(len(unique_fps), unique_fps.shape[1]))
    fp_pca.fit(unique_fps)
    v = fp_pca.components_[0]   # (H,)  first PC of fixed points

    # Orient v so that lower reward_site_idx → lower projection
    events_orient = find_pre_odor_onset_states(traj_data)
    _proj = events_orient['hidden'][:, 0] @ v
    if np.corrcoef(_proj, events_orient['reward_site'])[0, 1] < 0:
        v = -v

    # Input Jacobian
    def _jac_fn(h_prev, u_vec):
        def f(u):
            return network.apply(
                params, h_prev[None], u[None],
                method=lambda m, h, x: m.rnn_actor(h, x)
            )[0][0]
        return jax.jacfwd(f)(u_vec)

    contexts = [
        ('obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        ('obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(figsize[0] * 3, figsize[1] * 2))
    slopes = np.full((2, 3), np.nan)
    ratios = np.full((2, 3), np.nan)

    events         = find_pre_odor_onset_states(traj_data)
    sub_hidden_all = events['hidden'][:, 0]
    if color_by == 'inter_odor_site_distances':
        sub_color_all = events['isi']
    else:
        _spe3 = 20000
        _ne3  = max(1, traj_data[color_by].shape[0] // _spe3)
        color_raw     = traj_data[color_by].reshape(_ne3, _spe3, -1)[:, :, 0].T
        sub_color_all = color_raw[events['pre_t_idx'], events['trial_idx']]

    for row, (ctx_label, u_ctx, u_diff) in enumerate(contexts):
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))

        sub_hidden = sub_hidden_all.copy()
        sub_color  = sub_color_all.copy()

        if len(sub_hidden) == 0:
            print(f'No pre-odor onset events found for {ctx_label}')
            continue
        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // downsample), replace=False)
        sub_hidden = sub_hidden[keep]
        sub_color  = sub_color[keep]

        valid      = ~np.isnan(sub_color)
        sub_hidden = sub_hidden[valid]
        sub_color  = sub_color[valid]

        J_batch = np.array(_jac_batch(jnp.array(sub_hidden)))  # (N, H, input_dim)
        Ju      = J_batch @ np.array(u_diff)                   # (N, H)

        scalar_norm = np.linalg.norm(Ju, axis=1)              # (N,)
        scalar_sep  = (Ju @ w_hat) / np.maximum(scalar_norm, 1e-12)  # (N,100 / bottom-100) cosine
        sep_dist    = (sub_hidden @ w + b_diff) / w_norm      # (N,)
        x_fp        = sub_hidden @ v                           # (N,)

        for col, (scalar, ylabel, cmap_name) in enumerate([
            (scalar_norm, r'$\|J \cdot u_{\rm diff}\|$',              'viridis'),
            (scalar_sep,  r'$(Ju)^\top n_{\rm sep}\ /\ \|Ju\|$',     'coolwarm'),
            (sep_dist,    'Sep. distance',                             'coolwarm'),
        ]):
            ax = axes[row, col]
            if cmap_name == 'coolwarm':
                abs_max = max(float(np.abs(scalar).max()), 1e-8)
                norm = Normalize(vmin=-abs_max, vmax=abs_max)
            else:
                norm = Normalize(vmin=scalar.min(), vmax=scalar.max())
            cmap = plt.get_cmap(cmap_name)

            color_norm = Normalize(vmin=sub_color.min(), vmax=sub_color.max())
            sc = ax.scatter(x_fp, scalar, c=sub_color,
                            cmap='viridis', norm=color_norm,
                            s=6, alpha=1, linewidths=0)
            cb = plt.colorbar(sc, ax=ax, pad=0.02, label=_color_label)
            cb.solids.set_alpha(1)
            cb.outline.set_visible(False)
            n_tail = min(100, len(x_fp) // 2)
            order  = np.argsort(x_fp)
            tail_idx = np.concatenate([order[:n_tail], order[-n_tail:]])
            m, b = np.polyfit(x_fp[tail_idx], scalar[tail_idx], 1)
            slopes[row, col] = m
            mean_top = scalar[order[-n_tail:]].mean()
            mean_bot = scalar[order[:n_tail]].mean()
            ratios[row, col] = (mean_top - mean_bot) / mean_bot if mean_bot != 0 else np.nan
            x_line = np.array([x_fp.min(), x_fp.max()])
            ax.plot(x_line, m * x_line + b, color='red', linewidth=1.2, zorder=5)
            ax.set_xlabel('Projection onto FP PC1', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(ctx_label, fontsize=8)
            format_plot(ax)

    plt.tight_layout()
    return fig, slopes, ratios


# --- (from cell pre-patch-jac-2d-isi) ---
def plot_pre_patch_states_jac_2d_isi(traj_data, params, network,
                                      figsize=(3, 2.2), downsample=4,
                                      n_fp_attempts=200, fp_tol=1e-3):
    """
    X-axis : inter-odor site distance at onset
    Y-axis : |J @ u_diff|  or  (J @ u_diff)^T n_sep
    Color  : projection of the pre-patch state onto FP PC1

    States detected via find_pre_odor_onset_states (no in_patch filter).
    NaN ISI values are dropped.
    """

    # Separatrix normal
    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    w_hat  = w / np.linalg.norm(w)

    # Fixed-point first PC
    u_ctx_fp = jnp.array([1., 0., 0., 0., 0., 1., 0.])
    key      = jax.random.PRNGKey(42)
    indices  = jax.random.randint(key, (n_fp_attempts,), 0,
                                   traj_data['actor_hidden'].shape[0])
    h_inits  = jnp.array(traj_data['actor_hidden'][indices])
    fps, converged, _ = find_fixed_points_batch(
        params=params, network=network, input_vec=u_ctx_fp,
        h_inits=h_inits, max_steps=60000, learning_rate=0.001,
        tolerance=fp_tol, verbose=False,
    )
    unique_fps = np.array(filter_unique_fixed_points(fps, converged))
    assert len(unique_fps) > 0, "No fixed points found"
    fp_pca = PCA(n_components=min(len(unique_fps), unique_fps.shape[1]))
    fp_pca.fit(unique_fps)
    v = fp_pca.components_[0]

    def _jac_fn(h_prev, u_vec):
        def f(u):
            return network.apply(
                params, h_prev[None], u[None],
                method=lambda m, h, x: m.rnn_actor(h, x)
            )[0][0]
        return jax.jacfwd(f)(u_vec)

    contexts = [
        ('obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        ('obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[1] * 2))

    events         = find_pre_odor_onset_states(traj_data)
    sub_hidden_all = events['hidden'][:, 0]
    sub_isi_all    = events['isi']

    for row, (ctx_label, u_ctx, u_diff) in enumerate(contexts):
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))

        sub_hidden = sub_hidden_all.copy()
        sub_isi    = sub_isi_all.copy()

        if len(sub_hidden) == 0:
            print(f'No pre-odor onset events found for {ctx_label}')
            continue
        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // downsample), replace=False)
        sub_hidden = sub_hidden[keep]
        sub_isi    = sub_isi[keep]

        valid      = ~np.isnan(sub_isi)
        sub_hidden = sub_hidden[valid]
        sub_isi    = sub_isi[valid]

        if len(sub_hidden) == 0:
            print(f'All NaN for context {ctx_label}')
            continue

        J_batch = np.array(_jac_batch(jnp.array(sub_hidden)))
        Ju      = J_batch @ np.array(u_diff)

        scalar_norm = np.linalg.norm(Ju, axis=1)
        scalar_sep  = Ju @ w_hat
        fp_proj     = sub_hidden @ v

        fp_norm  = Normalize(vmin=fp_proj.min(), vmax=fp_proj.max())
        cmap     = plt.get_cmap('viridis')
        colors   = cmap(fp_norm(fp_proj))           # (N, 4) RGBA for edge colors
        jitter_rng = np.random.default_rng(1)
        x_jitter = sub_isi + jitter_rng.uniform(-0.3, 0.3, size=len(sub_isi))
        isi_ints = sub_isi.astype(int)
        unique_dists = np.unique(isi_ints)

        for col, (scalar, ylabel) in enumerate([
            (scalar_norm, r'$\|J \cdot u_{\rm diff}\|$'),
            (scalar_sep,  r'$(Ju)^\top n_{\rm sep}$'),
        ]):
            ax = axes[row, col]
            ax.scatter(x_jitter, scalar,
                       facecolors='none', edgecolors=colors,
                       s=18, alpha=0.4, linewidths=0.7)
            # Median line
            medians = [np.median(scalar[isi_ints == d]) for d in unique_dists]
            ax.plot(unique_dists, medians, color='k', linewidth=1.5, zorder=5)
            # Colorbar (manual, since we used edgecolors)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=fp_norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, pad=0.02, label='FP PC1 proj.')
            cb.outline.set_visible(False)
            ax.set_xlabel('Inter-odor site distance', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(ctx_label, fontsize=8)
            format_plot(ax)

    plt.tight_layout()
    return fig


def plot_pre_patch_states_jac_pc1_space(traj_data, params, network,
                                         figsize=(3, 2.2), downsample=4,
                                         n_fp_attempts=200, fp_tol=1e-3):
    """
    4×3 grid — rows 0-1: obs[2] / obs[3] Jacobian contexts;
               rows 2-3: direct orth-dist vs scalar plots.
    States detected via find_pre_odor_onset_states (in_patch filter applied).
    X-axis : FP PC1 projection.  Y-axis : orthogonal distance from FP PC1.
    """

    # Separatrix normal
    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    w_hat  = w / np.linalg.norm(w)

    # Fixed-point PC1
    u_ctx_fp = jnp.array([1., 0., 0., 0., 0., 1., 0.])
    key      = jax.random.PRNGKey(42)
    indices  = jax.random.randint(key, (n_fp_attempts,), 0,
                                   traj_data['actor_hidden'].shape[0])
    h_inits  = jnp.array(traj_data['actor_hidden'][indices])
    fps, converged, _ = find_fixed_points_batch(
        params=params, network=network, input_vec=u_ctx_fp,
        h_inits=h_inits, max_steps=60000, learning_rate=0.001,
        tolerance=fp_tol, verbose=False,
    )
    unique_fps = np.array(filter_unique_fixed_points(fps, converged))
    assert len(unique_fps) > 0, "No fixed points found"

    # Keep only stable fixed points (max |eigenvalue| <= 1)
    def _max_eig(fp):
        J = jax.jacfwd(lambda h: rnn_step_batch(
            h[None], u_ctx_fp, params, network).squeeze())(fp)
        return float(jnp.abs(jnp.linalg.eigvals(J)).max())

    stable_mask = np.array([_max_eig(jnp.array(fp)) <= 1.0 for fp in unique_fps])
    stable_fps  = unique_fps[stable_mask]
    assert len(stable_fps) > 0, "No stable fixed points found"
    print(f'  {stable_mask.sum()} / {len(unique_fps)} fixed points are stable')

    fp_pca   = PCA(n_components=min(len(stable_fps), stable_fps.shape[1])).fit(stable_fps)
    v        = fp_pca.components_[0]   # (H,)
    fp_mean  = fp_pca.mean_             # (H,)

    high_reward_site_index_h = traj_data['actor_hidden'][traj_data['reward_site_idx'] > 5][:100]
    low_reward_site_index_h  = traj_data['actor_hidden'][traj_data['reward_site_idx'] <= 5][:100]
    if np.mean((high_reward_site_index_h - low_reward_site_index_h) @ v) < 0:
        v = -v

    def _orth_dist_fp(h):
        c    = h - fp_mean
        proj = (c @ v)[:, None] * v[None, :]
        return np.linalg.norm(c - proj, axis=1)

    # Input Jacobian helper
    def _jac_fn(h_prev, u_vec):
        def f(u):
            return network.apply(
                params, h_prev[None], u[None],
                method=lambda m, h, x: m.rnn_actor(h, x)
            )[0][0]
        return jax.jacfwd(f)(u_vec)

    contexts = [
        ('obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        ('obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                         jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(figsize[0] * 3, figsize[1] * 4))
    slopes = np.full((2, 3), np.nan)

    events         = find_pre_odor_onset_states(traj_data)
    sub_hidden_all = events['hidden'][:, 0]
    sub_isi_all    = events['isi']

    for row, (ctx_label, u_ctx, u_diff) in enumerate(contexts):
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))

        sub_hidden = sub_hidden_all.copy()
        sub_isi    = sub_isi_all.copy()

        if len(sub_hidden) == 0:
            print(f'No pre-odor onset events found for {ctx_label}')
            continue
        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // downsample),
                          replace=False)
        sub_hidden = sub_hidden[keep]
        sub_isi    = sub_isi[keep]

        J_batch     = np.array(_jac_batch(jnp.array(sub_hidden)))  # (N, H, input_dim)
        Ju          = J_batch @ np.array(u_diff)                   # (N, H)
        scalar_norm = np.linalg.norm(Ju, axis=1)                   # (N,)
        scalar_sep  = Ju @ w_hat                                   # (N,)

        x_fp   = (sub_hidden - fp_mean) @ v   # PC1 projection (centred)
        y_orth = _orth_dist_fp(sub_hidden)     # orthogonal distance

        valid_isi = ~np.isnan(sub_isi)

        for col, (scalar, clabel, cmap_name, mask, vmin_mode) in enumerate([
            (scalar_norm, r'$\|J \cdot u_{\rm diff}\|$', 'viridis',  np.ones(len(sub_hidden), bool), 'data'),
            (scalar_sep,  r'$(Ju)^\top n_{\rm sep}$',    'coolwarm', np.ones(len(sub_hidden), bool), 'data'),
            (sub_isi,     'Inter-odor site dist.',        'viridis',  valid_isi,                      'data'),
        ]):
            ax = axes[row, col]
            cmap  = plt.get_cmap(cmap_name)
            s_masked = scalar[mask]
            if vmin_mode == 'zero':
                cnorm = Normalize(vmin=0, vmax=max(float(s_masked.max()), 1e-8))
            else:
                cnorm = Normalize(vmin=np.nanmin(s_masked), vmax=np.nanmax(s_masked))

            sc = ax.scatter(x_fp[mask], y_orth[mask], c=s_masked, cmap=cmap, norm=cnorm,
                            s=2, alpha=1, linewidths=0)
            cb = plt.colorbar(sc, ax=ax, pad=0.02, label=clabel)
            cb.solids.set_alpha(1)
            cb.outline.set_visible(False)
            ax.set_xlabel('FP PC1 proj.', fontsize=8)
            ax.set_ylabel('Dist. from FP PC1', fontsize=8)
            ax.set_title(ctx_label, fontsize=8)
            format_plot(ax)

            # Direct plot: orth dist vs metric
            ax2 = axes[row + 2, col]
            x2 = y_orth[mask]
            y2 = s_masked
            ax2.scatter(x2, y2, c='black', s=2, alpha=1, linewidths=0)
            m2, b2 = np.polyfit(x2, y2, 1)
            slopes[row, col] = m2
            x2_line = np.array([x2.min(), x2.max()])
            ax2.plot(x2_line, m2 * x2_line + b2, color='red', linewidth=1)
            ax2.set_xlabel('Dist. from FP PC1', fontsize=8)
            ax2.set_ylabel(clabel, fontsize=8)
            ax2.set_title(ctx_label, fontsize=8)
            format_plot(ax2)

    plt.tight_layout()
    return fig, slopes


# --- (from cell neuron-reward-corr) ---
def plot_neuron_reward_correlations(traj_data, top_n=5, figsize=(5, 3)):
    """
    For each neuron, compute the Pearson r between its activity and
    rewards_seen_in_patch across all time steps and trials.

    Produces:
      - A histogram of all neuron-reward correlations.
      - A scatter of activity vs reward for the top_n most-correlated neurons.
    """
    from scipy.stats import pearsonr

    _steps_per_env = 20000
    _n_envs = max(1, traj_data['actor_hidden'].shape[0] // _steps_per_env)
    hidden  = traj_data['actor_hidden'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)  # (T, n_trials, H)
    rewards = traj_data['rewards_seen_in_patch'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)  # (T, n_trials, ...)

    H = hidden.shape[2]
    reward_flat = rewards[..., 0].ravel()   # (T*n_trials,)
    hidden_flat = hidden.reshape(-1, H)     # (T*n_trials, H)

    rs = np.array([pearsonr(hidden_flat[:, h], reward_flat)[0] for h in range(H)])

    # --- Histogram ---
    fig1, ax = plt.subplots(figsize=figsize)
    ax.hist(rs, bins=30, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Pearson r (neuron activity vs rewards_seen_in_patch)')
    ax.set_ylabel('Neuron count')
    ax.set_title('Neuron–reward correlation distribution')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    fig1.tight_layout()

    # --- Scatter for top_n neurons ---
    top_idx = np.argsort(np.abs(rs))[::-1][:top_n]
    ncols = min(top_n, 5)
    fig2, axes = plt.subplots(1, ncols, figsize=(figsize[0] * ncols / 2, figsize[1]), sharey=False)
    if ncols == 1:
        axes = [axes]
    # Subsample for speed
    rng   = np.random.default_rng(0)
    n_pts = min(2000, len(reward_flat))
    sel   = rng.choice(len(reward_flat), size=n_pts, replace=False)
    for ax, h in zip(axes, top_idx):
        ax.scatter(reward_flat[sel], hidden_flat[sel, h], s=2, alpha=0.3, rasterized=True)
        ax.set_title(f'unit {h}\nr={rs[h]:.2f}', fontsize=9)
        ax.set_xlabel('rewards in patch')
        if ax is axes[0]:
            ax.set_ylabel('activity')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    fig2.suptitle(f'Top {top_n} neurons by |r|', y=1.02)
    fig2.tight_layout()

    return rs, fig1, fig2


# --- (from cell pre-patch-with-next) ---
def plot_pre_patch_states_with_next_3d(traj_data, params, pca,
                                        figsize=(5, 4), color_by='time', cbar_label='Time',
                                        elev=20, azim=30, clean=False):
    """
    States detected via find_pre_odor_onset_states.

    Returns (fig1, fig2):
      fig1 — 3-D scatter with trajectory lines through the next 4 steps.
      fig2 — 2-D: sep dist (initial and after 4 steps) vs pre-odor PC1.
    """
    events = find_pre_odor_onset_states(traj_data, max_lookahead=4)

    sub_hidden = events['hidden'][:, 0]
    sub_next1  = events['hidden'][:, 1]
    sub_next2  = events['hidden'][:, 2]
    sub_next3  = events['hidden'][:, 3]
    sub_next4  = events['hidden'][:, 4]

    if color_by == 'inter_odor_site_distances':
        sub_colors = events['isi']
    elif color_by == 'reward_site_idx':
        sub_colors = events['reward_site']
    elif color_by != 'time':
        _spe5 = 20000
        _ne5  = max(1, traj_data[color_by].shape[0] // _spe5)
        raw        = traj_data[color_by].reshape(_ne5, _spe5, -1)[:, :, 0].T
        sub_colors = raw[events['pre_t_idx'], events['trial_idx']]
    else:
        sub_colors = events['pre_t_idx'].astype(float)

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    # Patch number for each event
    _spe4 = 20000
    _ne4  = max(1, traj_data['current_patch_num'].shape[0] // _spe4)
    raw_patch      = traj_data['current_patch_num'].reshape(_ne4, _spe4, -1).transpose(1, 0, 2)[:, :, 0]
    event_patch    = raw_patch[events['pre_t_idx'], events['trial_idx']].astype(int)
    patch_vals     = sorted(np.unique(event_patch))
    n_patches      = len(patch_vals)

    d_pr_all    = participation_ratio(traj_data['actor_hidden'])
    d_pr_sub    = participation_ratio(sub_hidden) if len(sub_hidden) > 1 else float('nan')
    cmap        = plt.cm.viridis
    norm        = Normalize(vmin=np.nanmin(sub_colors), vmax=np.nanmax(sub_colors))

    def _add_lines(ax, p0, p1, p2, p3, p4, colors):
        segs = np.stack([p0, p1, p2, p3, p4], axis=1)
        lc = Line3DCollection(segs, colors=cmap(norm(colors)),
                              linewidths=0.6, alpha=0.5, zorder=1)
        ax.add_collection3d(lc)

    rng = np.random.default_rng(0)

    fig1 = plt.figure(figsize=(figsize[0] * n_patches, figsize[1]))
    fig2, axes2 = plt.subplots(1, n_patches,
                               figsize=(figsize[0] * n_patches, figsize[1]),
                               sharey=True)
    if n_patches == 1:
        axes2 = [axes2]

    ctx_pca = PCA(n_components=1).fit(sub_hidden)
    v = -ctx_pca.components_[0]

    for col, pnum in enumerate(patch_vals):
        mask = event_patch == pnum
        h0 = sub_hidden[mask]; h1 = sub_next1[mask]; h2 = sub_next2[mask]
        h3 = sub_next3[mask];  h4 = sub_next4[mask]
        sc = sub_colors[mask]
        panel_title = (f'Patch {pnum} — pre-site states\n'
                       f'$D_{{PR}}$ all={d_pr_all:.1f}  sub={participation_ratio(h0) if len(h0) > 1 else float("nan"):.1f}')

        # 3-D panel
        keep = rng.choice(len(h0), size=max(1, len(h0) // 40), replace=False)
        ax3  = fig1.add_subplot(1, n_patches, col + 1, projection='3d')
        ax3.set_title(panel_title, fontsize=8)
        pts_pre = pca.transform(h0[keep])[:, :3]
        pts_n1  = pca.transform(h1[keep])[:, :3]
        pts_n2  = pca.transform(h2[keep])[:, :3]
        pts_n3  = pca.transform(h3[keep])[:, :3]
        pts_n4  = pca.transform(h4[keep])[:, :3]
        _add_lines(ax3, pts_pre, pts_n1, pts_n2, pts_n3, pts_n4, sc[keep])
        ax3.scatter(pts_pre[:, 0], pts_pre[:, 1], pts_pre[:, 2],
                    c=sc[keep], cmap=cmap, norm=norm, s=12, alpha=0.6, depthshade=True)
        if clean:
            ax3.set_axis_off()
        else:
            _style_3d_ax(ax3, norm, cmap, cbar_label)
        add_separatrix_plane(ax3, params, pca)
        ax3.view_init(elev=elev, azim=azim)

        # 2-D panel
        ax2 = axes2[col]
        x   = h0 @ v
        d0  = sep_dist(h0)
        d3  = sep_dist(h3)
        ax2.scatter(x, d0, color='steelblue', s=2, alpha=1, linewidths=0, label='initial')
        ax2.scatter(x, d3, color='tomato',    s=2, alpha=1, linewidths=0, label='after 4 steps')
        ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Projection onto pre-odor PC1', fontsize=8)
        if col == 0:
            ax2.set_ylabel('Sep. distance', fontsize=8)
        ax2.set_title(panel_title, fontsize=8)
        ax2.legend(fontsize=7, frameon=False)
        format_plot(ax2)

    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2


def plot_pre_patch_states_with_next_2d(traj_data, params, figsize=(5, 4), decoder_dir=None):
    """
    2-D: open circles = initial hidden state, solid = after 3 steps.
    Coloured by ISI (viridis). NaN ISI dropped.
    States detected via find_pre_odor_onset_states (in_patch filter).
    x-axis: pre-odor PC1.  y-axis: sep. distance.
    """
    events = find_pre_odor_onset_states(traj_data)
    mask   = ~np.isnan(events['isi'])
    if not mask.any():
        print('No valid pre-odor onset events')
        return None

    sub_hidden = events['hidden'][mask, 0]
    sub_next3  = events['hidden'][mask, 3]
    sub_isi    = events['isi'][mask]

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    if decoder_dir is None:
        decoder_dir = fit_remaining_time_decoder(traj_data)
    v  = decoder_dir
    x  = sub_hidden @ v
    d0 = sep_dist(sub_hidden)
    d3 = sep_dist(sub_next3)

    norm   = Normalize(vmin=np.nanmin(sub_isi), vmax=np.nanmax(sub_isi))
    cmap   = plt.cm.viridis
    colors = cmap(norm(sub_isi))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, d0, facecolors='none', edgecolors=colors,
               s=10, alpha=1, linewidths=0.8, label='initial')
    ax.scatter(x, d3, c=sub_isi, cmap=cmap, norm=norm,
               s=10, alpha=1, linewidths=0, label='after 3 steps')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, label='Inter-odor site distance')
    cb.outline.set_visible(False)

    ax.set_xlabel('Remaining time proj.', fontsize=8)
    ax.set_ylabel('Sep. distance', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    format_plot(ax)
    fig.tight_layout()
    return fig


def plot_pre_patch_states_fp_pc1_2d(traj_data, params, network,
                                     figsize=(5, 4), n_fp_attempts=200, fp_tol=1e-3):
    """
    x-axis: projection onto stable FP PC1 (centred at FP mean).
    y-axis: signed separatrix distance.
    Coloured by ISI (viridis).  open = initial, solid = after 3 steps.
    States detected via find_pre_odor_onset_states (in_patch filter).
    """
    events = find_pre_odor_onset_states(traj_data)
    if len(events['hidden']) == 0:
        print('No pre-odor onset events found')
        return None

    sub_hidden = events['hidden'][:, 0]
    sub_next3  = events['hidden'][:, 3]
    sub_site   = events['isi']

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    u_ctx_fp = jnp.array([1., 0., 0., 0., 0., 1., 0.])
    key      = jax.random.PRNGKey(42)
    indices  = jax.random.randint(key, (n_fp_attempts,), 0,
                                   traj_data['actor_hidden'].shape[0])
    h_inits  = jnp.array(traj_data['actor_hidden'][indices])
    fps, converged, _ = find_fixed_points_batch(
        params=params, network=network, input_vec=u_ctx_fp,
        h_inits=h_inits, max_steps=60000, learning_rate=0.001,
        tolerance=fp_tol, verbose=False,
    )
    unique_fps = np.array(filter_unique_fixed_points(fps, converged))
    assert len(unique_fps) > 0, "No fixed points found for FP PC1"

    def _max_eig(fp):
        J = jax.jacfwd(lambda h: rnn_step_batch(
            h[None], u_ctx_fp, params, network).squeeze())(fp)
        return float(jnp.abs(jnp.linalg.eigvals(J)).max())

    stable_mask = np.array([_max_eig(jnp.array(fp)) <= 1.0 for fp in unique_fps])
    stable_fps  = unique_fps[stable_mask]
    assert len(stable_fps) > 0, "No stable fixed points found"
    print(f'  {stable_mask.sum()} / {len(unique_fps)} fixed points stable')

    fp_pca  = PCA(n_components=1).fit(stable_fps)
    v       = -fp_pca.components_[0]
    fp_mean = fp_pca.mean_

    x0 = (sub_hidden - fp_mean) @ v
    x3 = (sub_next3  - fp_mean) @ v
    d0 = sep_dist(sub_hidden)
    d3 = sep_dist(sub_next3)

    norm   = Normalize(vmin=np.nanmin(sub_site), vmax=np.nanmax(sub_site))
    cmap   = plt.cm.viridis
    colors = cmap(norm(sub_site))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x0, d0, facecolors='none', edgecolors=colors,
               s=5, alpha=1, linewidths=0.8, label='initial')
    ax.scatter(x3, d3, c=sub_site, cmap=cmap, norm=norm,
               s=5, alpha=1, linewidths=0, label='after 3 steps')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, label='ISI')
    cb.outline.set_visible(False)
    ax.set_xlabel('Projection onto FP PC1', fontsize=8)
    ax.set_ylabel('Sep. distance', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    format_plot(ax)
    fig.tight_layout()
    return fig


def plot_sep_dist_change_vs_isi(traj_data, params, figsize=(5, 4), n_groups=5, decoder_dir=None):
    """
    States detected via find_pre_odor_onset_states (in_patch filter).

    Returns (fig1, fig2, fig3, fig4):
      fig1 — ISI vs Δsep_dist, coloured by initial separatrix distance (coolwarm).
      fig2 — ISI vs pre-odor PC1 projection, coloured by final sep. distance (coolwarm).
      fig3 — ISI vs final sep. distance, N line curves conditioned on PC1-projection
              groups (equal-width bins); lines coloured by group using plasma colormap.
      fig4 — same as fig3 but y-axis is change in sep. distance (Δ = final − initial).
    """
    events = find_pre_odor_onset_states(traj_data)
    if len(events['hidden']) == 0:
        print('No pre-odor onset events found')
        return None, None, None, None

    sub_hidden = events['hidden'][:, 0]
    sub_next3  = events['hidden'][:, 3]
    sub_site   = events['isi']

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    d0      = sep_dist(sub_hidden)
    d3      = sep_dist(sub_next3)
    delta_d = d3 - d0
    abs_max = max(float(np.abs(d3).max()), 1e-8)
    d0_norm = Normalize(vmin=-max(float(np.abs(d0).max()), 1e-8),
                        vmax= max(float(np.abs(d0).max()), 1e-8))
    d3_norm = Normalize(vmin=-abs_max, vmax=abs_max)

    fig1, ax1 = plt.subplots(figsize=figsize)
    sc1 = ax1.scatter(sub_site, delta_d, c=d0, cmap='coolwarm', norm=d0_norm,
                      s=5, alpha=0.6, linewidths=0)
    cb1 = plt.colorbar(sc1, ax=ax1, pad=0.02, label='Initial sep. distance')
    cb1.outline.set_visible(False)
    ax1.set_xlabel('Inter reward site distance', fontsize=8)
    ax1.set_ylabel('Movement ortho to sep.', fontsize=8)
    format_plot(ax1)
    fig1.tight_layout()

    if decoder_dir is None:
        decoder_dir = fit_remaining_time_decoder(traj_data)
    pc1 = sub_hidden @ decoder_dir
    jitter = np.random.default_rng(0).normal(scale=0.1, size=sub_site.shape)
    fig2, ax2 = plt.subplots(figsize=figsize)
    sc2 = ax2.scatter(sub_site + jitter, pc1, c=d3, cmap='coolwarm', norm=d3_norm,
                      s=5, alpha=1, linewidths=0)
    cb2 = plt.colorbar(sc2, ax=ax2, pad=0.02, label='Final sep. distance')
    cb2.outline.set_visible(False)
    ax2.set_xlabel('Inter reward site distance', fontsize=8)
    ax2.set_ylabel('Remaining time proj.', fontsize=8)
    format_plot(ax2)
    fig2.tight_layout()

    pc1_min, pc1_max = float(pc1.min()), float(pc1.max())
    group_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_groups))
    fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
    for j in range(n_groups):
        lo = pc1_min + (pc1_max - pc1_min) * j / n_groups
        hi = pc1_min + (pc1_max - pc1_min) * (j + 1) / n_groups
        mask_g = (pc1 >= lo) & (pc1 <= hi if j == n_groups - 1 else pc1 < hi)
        if not mask_g.any():
            continue
        isi_g = sub_site[mask_g]
        d3_g  = d3[mask_g]
        unique_isi = np.unique(isi_g)
        vals       = [d3_g[isi_g == u] for u in unique_isi]
        mean_d3    = np.array([v.mean() for v in vals])
        sem_d3     = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                               for v in vals])
        ax3.plot(unique_isi, mean_d3, color=group_colors[j], linewidth=1.5)
        ax3.fill_between(unique_isi, mean_d3 - sem_d3, mean_d3 + sem_d3,
                         color=group_colors[j], alpha=0.25, linewidth=0)
    ax3.set_xticks(np.unique(sub_site))
    ax3.axhline(0, color='k', linewidth=0.8, linestyle='--')

    sm3 = plt.cm.ScalarMappable(
        cmap='plasma', norm=Normalize(vmin=pc1_min, vmax=pc1_max))
    sm3.set_array([])
    cb3 = plt.colorbar(sm3, ax=ax3, pad=0.02, label='Remaining time proj.')
    cb3.outline.set_visible(False)
    ax3.set_xlabel('Inter reward site distance', fontsize=8)
    ax3.set_ylabel('Final sep. distance', fontsize=8)
    format_plot(ax3)
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(figsize=(3.5, 2.5))
    for j in range(n_groups):
        lo = pc1_min + (pc1_max - pc1_min) * j / n_groups
        hi = pc1_min + (pc1_max - pc1_min) * (j + 1) / n_groups
        mask_g = (pc1 >= lo) & (pc1 <= hi if j == n_groups - 1 else pc1 < hi)
        if not mask_g.any():
            continue
        isi_g    = sub_site[mask_g]
        delta_g  = delta_d[mask_g]
        unique_isi = np.unique(isi_g)
        vals       = [delta_g[isi_g == u] for u in unique_isi]
        mean_dd    = np.array([v.mean() for v in vals])
        sem_dd     = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                               for v in vals])
        ax4.plot(unique_isi, mean_dd, color=group_colors[j], linewidth=1.5)
        ax4.fill_between(unique_isi, mean_dd - sem_dd, mean_dd + sem_dd,
                         color=group_colors[j], alpha=0.25, linewidth=0)
    ax4.set_xticks(np.unique(sub_site))
    sm4 = plt.cm.ScalarMappable(
        cmap='plasma', norm=Normalize(vmin=pc1_min, vmax=pc1_max))
    sm4.set_array([])
    cb4 = plt.colorbar(sm4, ax=ax4, pad=0.02, label='Remaining time proj.')
    cb4.outline.set_visible(False)
    ax4.set_xlabel('Inter reward site distance', fontsize=8)
    ax4.set_ylabel(r'$\Delta$ sep. distance ', fontsize=8)
    format_plot(ax4)
    fig4.tight_layout()
    return fig1, fig2, fig3, fig4


def compute_post_odor_trajectories(traj_data, params, network, max_n=5, tte_data=None, patch_num=None):
    """
    Collect post-odor onset hidden states, run them forward under input A, and compute
    a scalar characterisation per state used for binning/coloring throughout the plots.

    Priority for the characterisation axis:
      1. tte_data provided  → KNN time-to-exit (knn_time_to_exit)
      2. decoder_dir provided → linear projection onto decoder_dir
      3. neither             → PCA PC1 of pre-odor states

    Returns dict with keys:
      h_inits     (N, H)  — post-odor onset states
      trajs_a     (max_n+1, N, H) — states after 0..max_n A-steps
      pca_pre     — PCA(1) fitted on pre-odor states (None if tte_data/decoder_dir provided)
      pc1_init    (N,)  — characterisation value for h_inits
      pc1_by_n    list of (N,) — characterisation after each n in 1..max_n A-steps
      n_vals      list of ints 1..max_n
      decoder_dir (H,) or None
      proj_label  str  — label for axis/colorbar
    """
    input_a    = jnp.array([1., 0., 0., 0., 0., 1., 0.])
    _steps_per_env = 20000
    _n_envs = max(1, traj_data['actor_hidden'].shape[0] // _steps_per_env)
    hidden_all = traj_data['actor_hidden'].reshape(_n_envs, _steps_per_env, -1).transpose(1, 0, 2)
    rsi_all    = traj_data['reward_site_idx'].reshape(_n_envs, _steps_per_env).T
    _, n_trials, _ = hidden_all.shape

    h_list = []
    rsi_list = []
    for trial in range(n_trials):
        diffs   = np.diff(rsi_all[:, trial])
        valid_t = np.where((diffs <= -1))[0]
        for t in valid_t:
            if t + 2 < hidden_all.shape[0]:
                h_list.append(hidden_all[t + 2, trial])
                rsi_list.append(rsi_all[t + 1, trial])

    if len(h_list) == 0:
        return None

    h_inits = np.array(h_list)
    trajs_a = np.array(run_states_forward(
        jnp.array(h_inits), input_a, params, network, n_steps=max_n + 1))

    n_vals = list(range(1, max_n + 1))

    knn_tte_init   = patch_progress(h_inits, tte_data)
    knn_tte_by_n   = [patch_progress(np.array(trajs_a[n]), tte_data) for n in n_vals]
    proj_label = 'Patch progress'

    return dict(h_inits=h_inits, trajs_a=trajs_a,
                knn_tte_init=knn_tte_init, knn_tte_by_n=knn_tte_by_n,
                n_vals=n_vals, proj_label=proj_label)


def plot_post_odor_sep_dist_vs_steps(traj_data, params, network, precomputed=None, n_groups=10, scat_ylim=None, figsize=(5, 4), group_min=20):
    """
    Collects all post-odor hidden states (reward_site_idx > 0, diff >= -1).
    For each state and n in {1..5}:
      - runs forward n steps with input A = [1,0,0,0,0,1,0]
      - then 3 steps with input B = [1,0,0,1,0,1,0]
      - measures sep distance of the resulting state
    PC1 axis is fitted on actual pre-odor states (find_pre_odor_onset_states hidden[:,0]);
    projections and groupings use this axis throughout.

    Returns (fig_final, fig_delta, fig_final_scatter, fig_delta_scatter, fig_pc1_move):
      fig_final         — n-steps vs final sep. distance, lines conditioned on before-B PC1 group.
      fig_delta         — n-steps vs Δsep. distance (final − before-B), lines conditioned on before-B PC1 group.
      fig_final_scatter — same data as scatter, coloured continuously by before-B PC1.
      fig_delta_scatter — same data as scatter, coloured continuously by before-B PC1.
      fig_pc1_move      — n-steps vs Δ PC1 projection during A steps, coloured by initial PC1 proj.
      fig_pc1_n5        — n=5 condition: initial PC1 proj. (x) vs Δ PC1 after 5 A steps (y).
      fig_final_by_a    — same as fig_final but binned by PC1 projection after all A steps.
      fig_delta_by_a    — same as fig_delta but binned by PC1 projection after all A steps.
    """
    input_b      = jnp.array([1., 0., 0., 1., 0., 1., 0.])
    max_n        = 5
    n_steps_odor = 6
    site_length  = 3

    if precomputed is None:
        precomputed = compute_post_odor_trajectories(traj_data, params, network, max_n=max_n)
    if precomputed is None:
        print('No post-odor states found')
        return None, None, None, None, None

    h_inits    = precomputed['h_inits']
    trajs_a    = precomputed['trajs_a']
    knn_tte_init   = precomputed['knn_tte_init']
    knn_tte_by_n   = precomputed['knn_tte_by_n']
    n_vals     = precomputed['n_vals']
    proj_label = precomputed.get('proj_label', 'Pre-odor PC1 proj.')

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    d_final_by_n       = []
    d_before_odor_by_n = []
    stops_by_n         = []
    stop_counts_by_n   = []
    for n in n_vals:
        h_before_odor     = np.array(trajs_a[n])
        trajs_before_odor = np.array(run_states_forward(
            jnp.array(h_before_odor), input_b, params, network,
            n_steps=n_steps_odor + 1, use_self_action=True))
        d_final_by_n.append(sep_dist(np.array(trajs_before_odor[site_length])))
        d_before_odor_by_n.append(sep_dist(h_before_odor))
        stop_counts = np.sum(
            np.stack([sep_dist(np.array(trajs_before_odor[t])) > 0
                      for t in range(1, n_steps_odor + 1)], axis=0).astype(float),
            axis=0)
        stops_by_n.append((stop_counts >= 3).astype(float))
        stop_counts_by_n.append(stop_counts)
    knn_tte_min         = float(knn_tte_init.min())
    knn_tte_max         = float(knn_tte_init.max())
    _div_cmap    = plt.cm.coolwarm
    group_colors = _div_cmap(np.linspace(0.05, 0.95, n_groups))

    def _lines_fig(y_by_n, ylabel, hline=False,
                   pc1_group=None, gmin=None, gmax=None, cbar_label=None,
                   pc1_group_by_n=None):
        if cbar_label is None:
            cbar_label = proj_label
        # pc1_group_by_n: list of per-n arrays; overrides pc1_group for dynamic binning
        if pc1_group_by_n is None:
            if pc1_group is None:
                pc1_group, gmin, gmax = knn_tte_init, knn_tte_min, knn_tte_max
        else:
            all_vals = np.concatenate(pc1_group_by_n)
            gmin, gmax = float(all_vals.min()), float(all_vals.max())
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for j in range(n_groups):
            lo    = gmin + (gmax - gmin) * j / n_groups
            hi    = gmin + (gmax - gmin) * (j + 1) / n_groups
            means, sems = [], []
            for i, y in enumerate(y_by_n):
                grp   = pc1_group_by_n[i] if pc1_group_by_n is not None else pc1_group
                mask_g = (grp >= lo) & (grp <= hi if j == n_groups - 1 else grp < hi)
                # downsample mask_g to 10 points
                if mask_g.sum() < group_min:
                    mask_g = None
                else:
                    # sample 10 nonzero points from mask_g without replacement
                    mask_g_nonzero_idx = np.arange(len(mask_g))[mask_g]
                    np.random.shuffle(mask_g_nonzero_idx)
                    mask_g_nonzero_idx_clipped = mask_g_nonzero_idx[:group_min]
                    mask_g = np.zeros_like(mask_g, dtype=bool)
                    mask_g[mask_g_nonzero_idx_clipped] = True
                if mask_g is None or not mask_g.any():
                    means.append(np.nan); sems.append(0.)
                else:
                    means.append(y[mask_g].mean())
                    sems.append(y[mask_g].std(ddof=1) / np.sqrt(mask_g.sum())
                                if mask_g.sum() > 1 else 0.)
            means, sems = np.array(means), np.array(sems)
            if np.all(np.isnan(means)):
                continue
            ax.plot(n_vals, means, color=group_colors[j], linewidth=1.5)
            ax.fill_between(n_vals, means - sems, means + sems,
                            color=group_colors[j], alpha=0.25, linewidth=0)
        # overall_means = np.array([y.mean() for y in y_by_n])
        # overall_sems  = np.array([y.std(ddof=1) / np.sqrt(len(y)) for y in y_by_n])
        # ax.plot(n_vals, overall_means, color='k', linewidth=2, zorder=5)
        # ax.fill_between(n_vals, overall_means - overall_sems, overall_means + overall_sems,
        #                 color='k', alpha=0.15, linewidth=0, zorder=4)
        if hline:
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        sm = plt.cm.ScalarMappable(
            cmap=_div_cmap, norm=Normalize(vmin=gmin, vmax=gmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, label=cbar_label)
        cb.outline.set_visible(False)
        ax.set_xticks(n_vals)
        ax.set_xlabel('Steps before odor', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        format_plot(ax)
        fig.tight_layout()
        return fig

    fig_final  = _lines_fig(d_final_by_n, 'Final sep. distance', hline=True)
    delta_by_n = [d_f - d_b for d_f, d_b in zip(d_final_by_n, d_before_odor_by_n)]
    fig_delta  = _lines_fig(delta_by_n, r'$\Delta$ sep. distance')

    fig_final_by_a = _lines_fig(d_final_by_n, 'Final sep. distance',
                                pc1_group_by_n=knn_tte_by_n,
                                cbar_label=f'{proj_label} (after n A steps)', hline=True)
    fig_delta_by_a = _lines_fig(delta_by_n, r'$\Delta$ sep. distance',
                                pc1_group_by_n=knn_tte_by_n,
                                cbar_label=f'{proj_label} (after pre-odor steps)')

    # Scatter versions: tile n_vals across all N states
    rng        = np.random.default_rng(0)
    sc_n       = np.repeat(n_vals, len(h_inits)).astype(float)
    sc_n      += rng.normal(scale=0.08, size=sc_n.shape)
    sc_knn_tte     = np.concatenate(knn_tte_by_n)
    sc_final   = np.concatenate(d_final_by_n)
    sc_delta   = np.concatenate(delta_by_n)
    knn_tte_norm   = Normalize(vmin=knn_tte_min, vmax=knn_tte_max)

    # Downsample to at most 5000 points for scatter plots
    _sc_rng  = np.random.default_rng(1)
    _sc_idx  = _sc_rng.choice(len(sc_n), size=min(2000, len(sc_n)), replace=False)
    _sc_idx  = np.sort(_sc_idx)

    def _scatter_fig(sc_y, ylabel, hline=False, norm=knn_tte_norm, c=sc_knn_tte):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sc = ax.scatter(sc_n[_sc_idx], sc_y[_sc_idx], c=c[_sc_idx],
                        cmap=_div_cmap, norm=norm, s=3, alpha=1, linewidths=0)
        if hline:
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        cb = plt.colorbar(sc, ax=ax, pad=0.02, label=proj_label)
        cb.outline.set_visible(False)
        ax.set_xticks(n_vals)
        ax.set_xlabel('Steps before odor', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        format_plot(ax)
        fig.tight_layout()
        return fig

    fig_final_scatter = _scatter_fig(sc_final, 'Final sep. distance', hline=True)
    fig_delta_scatter = _scatter_fig(sc_delta, r'$\Delta$ sep. distance')

    # KNN TTE movement during A steps: coloured by initial KNN TTE projection (same axis/cmap as first 4)
    sc_knn_tte_move  = np.concatenate([knn_tte_n - knn_tte_init for knn_tte_n in knn_tte_by_n])
    sc_knn_tte_color = np.tile(knn_tte_init, len(n_vals))

    fig_knn_tte_move, ax_pm = plt.subplots(figsize=(3.5, 2.5))
    sc_pm = ax_pm.scatter(sc_n[_sc_idx], sc_knn_tte_move[_sc_idx],
                          c=sc_knn_tte_color[_sc_idx], cmap=_div_cmap,
                          norm=knn_tte_norm, s=3, alpha=1, linewidths=0)
    ax_pm.axhline(0, color='k', linewidth=0.8, linestyle='--')
    cb_pm = plt.colorbar(sc_pm, ax=ax_pm, pad=0.02, label=f'Initial patch progress')
    cb_pm.outline.set_visible(False)
    ax_pm.set_xticks(n_vals)
    ax_pm.set_xlabel('Steps before odor', fontsize=8)
    ax_pm.set_ylabel(r'$\Delta$ proj.', fontsize=8)
    format_plot(ax_pm)
    fig_knn_tte_move.tight_layout()

    # n=5 condition: delta KNN TTE vs initial KNN TTE
    delta_knn_tte_n5 = knn_tte_by_n[4] - knn_tte_init
    rng_ds = np.random.default_rng(0)
    ds_idx = rng_ds.choice(len(knn_tte_init), size=int(0.2 * len(knn_tte_init)), replace=False)
    x_data = knn_tte_init[ds_idx]
    y_data = delta_knn_tte_n5[ds_idx]
    fig_knn_tte_n5, ax_n5 = plt.subplots(figsize=(3.5, 2.5))
    sns.kdeplot(x=x_data, y=y_data, ax=ax_n5,
                fill=True, levels=4, cmap='Blues', thresh=0.05, bw_adjust=1.5, gridsize=400)
    if scat_ylim is not None:
        ax_n5.set_ylim(*scat_ylim)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig_knn_tte_n5.colorbar(sm, ax=ax_n5, pad=0.02, label='Norm. density')
    cb.outline.set_visible(False)
    ax_n5.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax_n5.set_xlabel(f'Initial patch progress', fontsize=8)
    ax_n5.set_ylabel(r'$\Delta$ proj. (5 steps)', fontsize=8)
    format_plot(ax_n5)
    fig_knn_tte_n5.tight_layout()

    fig_stops_init = _lines_fig(
        stops_by_n, 'P(site stop)',
        cbar_label=f'Initial patch progress')
    fig_stops_after_a = _lines_fig(
        stops_by_n, 'P(site stop)',
        pc1_group_by_n=knn_tte_by_n, cbar_label=f'Patch progress after pre-odor steps')
    _n_offsets = np.linspace(-0.3, 0.3, len(n_vals))   # one y-offset per n value
    _sc_rng        = np.random.default_rng(0)
    sc_tte_after   = np.concatenate(knn_tte_by_n)
    sc_stop_counts = np.concatenate(stop_counts_by_n)
    sc_n_steps     = np.concatenate([np.full(len(v), n) for n, v in zip(n_vals, stop_counts_by_n)])
    sc_y           = np.concatenate([v + _n_offsets[i] for i, v in enumerate(stop_counts_by_n)])
    sc_y           = sc_y + _sc_rng.normal(0, 0.015, size=len(sc_y))

    # Within each TTE bin, equalise steps_before_odor distribution
    _n_tte_bins = 20
    _bin_edges  = np.linspace(sc_tte_after.min(), sc_tte_after.max(), _n_tte_bins + 1)
    _tte_bin    = np.clip(np.digitize(sc_tte_after, _bin_edges[:-1]) - 1, 0, _n_tte_bins - 1)
    _keep = []
    for _b in range(_n_tte_bins):
        _bin_mask      = _tte_bin == _b
        _n_idxs_in_bin = {n: np.where(_bin_mask & (sc_n_steps == n))[0] for n in n_vals}
        if any(len(v) == 0 for v in _n_idxs_in_bin.values()):
            continue
        _target = min(len(v) for v in _n_idxs_in_bin.values())
        for _idxs in _n_idxs_in_bin.values():
            _keep.extend(_sc_rng.choice(_idxs, size=_target, replace=False).tolist())
    _keep = np.array(_keep)

    fig_stop_counts_after_a, ax_sc = plt.subplots(figsize=(5.5, 5.0))
    _n_norm = Normalize(vmin=n_vals[0], vmax=n_vals[-1])
    sc_h = ax_sc.scatter(sc_tte_after[_keep], sc_y[_keep], c=sc_n_steps[_keep],
                         cmap='viridis', norm=_n_norm, s=3, alpha=1.0, linewidths=0)
    cb = fig_stop_counts_after_a.colorbar(sc_h, ax=ax_sc, pad=0.02, label='Steps before odor')
    cb.outline.set_visible(False)
    cb.set_ticks(n_vals)
    ax_sc.set_xlabel(f'{proj_label} after pre-odor steps', fontsize=8)
    ax_sc.set_ylabel('Steps stopped in odor', fontsize=8)
    format_plot(ax_sc)
    fig_stop_counts_after_a.tight_layout()

    # Bubble plot: one point per (tte_bin, steps_stopped, steps_before_odor) triplet
    # size = fraction of (tte_bin, n_steps) points with that stop count
    _bin_centers    = 0.5 * (_bin_edges[:-1] + _bin_edges[1:])
    _bin_width      = _bin_edges[1] - _bin_edges[0]
    _stop_vals      = np.arange(int(sc_stop_counts.max()) + 1)
    _n_offsets_bub  = np.linspace(-0.3, 0.3, len(n_vals))
    _bub_x, _bub_y, _bub_frac, _bub_n = [], [], [], []
    for _b in range(_n_tte_bins):
        _bin_mask = _tte_bin[_keep] == _b
        _pts_stop = sc_stop_counts[_keep][_bin_mask]
        _pts_n    = sc_n_steps[_keep][_bin_mask]
        if len(_pts_stop) == 0:
            continue
        for _ni, _n in enumerate(n_vals):
            _n_mask  = _pts_n == _n
            _n_count = _n_mask.sum()
            if _n_count == 0:
                continue
            for _sv in _stop_vals:
                _bub_x.append(_bin_centers[_b])
                _bub_y.append(_sv + _n_offsets_bub[_ni])
                _bub_frac.append(np.mean(_pts_stop[_n_mask] == _sv))
                _bub_n.append(_n)
    _bub_x    = np.array(_bub_x)
    _bub_y    = np.array(_bub_y)
    _bub_frac = np.array(_bub_frac)
    _bub_n    = np.array(_bub_n)

    fig_stop_bubble, ax_bub = plt.subplots(figsize=(6.0, 4.0))
    _max_size   = 200
    _n_norm_bub = Normalize(vmin=n_vals[0], vmax=n_vals[-1])
    sc_bub = ax_bub.scatter(_bub_x, _bub_y, s=_bub_frac * _max_size,
                            c=_bub_n, cmap='viridis', norm=_n_norm_bub,
                            alpha=1.0, linewidths=0)
    cb_bub = fig_stop_bubble.colorbar(sc_bub, ax=ax_bub, pad=0.02, label='Steps before odor')
    cb_bub.outline.set_visible(False)
    cb_bub.set_ticks(n_vals)
    for _leg_frac in [0.1, 0.3, 0.5]:
        ax_bub.scatter([], [], s=_leg_frac * _max_size, color='gray',
                       alpha=0.8, linewidths=0, label=f'{_leg_frac:.0%}')
    ax_bub.legend(title='Fraction', frameon=False, fontsize=7, title_fontsize=7,
                  loc='lower left')
    ax_bub.set_xlabel(f'{proj_label} after pre-odor steps', fontsize=8)
    ax_bub.set_ylabel('Steps stopped in odor', fontsize=8)
    ax_bub.set_yticks(_stop_vals)
    format_plot(ax_bub)
    fig_stop_bubble.tight_layout()

    # Marginal bubble plot: one point per (tte_bin, steps_stopped), collapsed over n_steps
    _mbub_x, _mbub_y, _mbub_frac = [], [], []
    for _b in range(_n_tte_bins):
        _bin_mask = _tte_bin[_keep] == _b
        _pts      = sc_stop_counts[_keep][_bin_mask]
        if len(_pts) == 0:
            continue
        for _sv in _stop_vals:
            _mbub_x.append(_bin_centers[_b])
            _mbub_y.append(_sv)
            _mbub_frac.append(np.mean(_pts == _sv))
    _mbub_x    = np.array(_mbub_x)
    _mbub_y    = np.array(_mbub_y)
    _mbub_frac = np.array(_mbub_frac)

    fig_stop_bubble_marginal, ax_mbub = plt.subplots(figsize=(6.0, 4.0))
    ax_mbub.scatter(_mbub_x, _mbub_y, s=_mbub_frac * _max_size,
                    color='steelblue', alpha=1.0, linewidths=0)
    for _leg_frac in [0.1, 0.3, 0.5]:
        ax_mbub.scatter([], [], s=_leg_frac * _max_size, color='steelblue',
                        alpha=1.0, linewidths=0, label=f'{_leg_frac:.0%}')
    ax_mbub.legend(title='Fraction', frameon=False, fontsize=7, title_fontsize=7,
                   loc='lower left')
    ax_mbub.set_xlabel(f'{proj_label} after pre-odor steps', fontsize=8)
    ax_mbub.set_ylabel('Steps stopped in odor', fontsize=8)
    ax_mbub.set_yticks(_stop_vals)
    format_plot(ax_mbub)
    fig_stop_bubble_marginal.tight_layout()

    return (fig_final, fig_delta, fig_final_scatter, fig_delta_scatter,
            fig_knn_tte_move, fig_knn_tte_n5, fig_final_by_a, fig_delta_by_a,
            fig_stops_init, fig_stops_after_a, fig_stop_counts_after_a,
            fig_stop_bubble, fig_stop_bubble_marginal)


def plot_post_odor_trajectories_pca(traj_data, params, network,
                                    precomputed=None, n_traj=25, figsize=(11, 9),
                                    elev1=20, azim1=30, elev2=20, azim2=120,
                                    tte_data=None, n_neighbors=10):
    """
    Visualise post-odor onset trajectories in the top 3 PCs of all network hidden states.
    Runs each trajectory forward the same way as plot_post_odor_sep_dist_vs_steps:
      n A-steps (input A, with self-action) then n_steps_odor odor steps (input B).
    Returns a list of 5 figures, one per n in {1..5} pre-odor A-steps.
    Each figure has two side-by-side 3D panels showing the same trajectories from
    different viewpoints (elev1/azim1 and elev2/azim2), plus a sep-distance panel below.
    Trajectories are coloured by KNN-predicted time-to-exit (tte_data provided) or
    by initial decoder_dir projection, falling back to tab10 if neither is available.
    Solid lines = A-steps; dashed lines = odor steps.
    """
    from matplotlib.lines import Line2D

    max_n        = 5
    n_steps_odor = 6
    input_b      = jnp.array([1., 0., 0., 1., 0., 1., 0.])

    if precomputed is None:
        precomputed = compute_post_odor_trajectories(traj_data, params, network, max_n=max_n)
    if precomputed is None:
        print('No post-odor states found')
        return [None] * max_n

    h_inits     = precomputed['h_inits']
    trajs_a     = precomputed['trajs_a']      # (max_n+1, N, H)
    n_vals      = precomputed['n_vals']       # [1..5]
    decoder_dir = precomputed.get('decoder_dir', None)
    proj_label  = precomputed.get('proj_label', 'Remaining time proj.')
    H           = h_inits.shape[-1]

    # Fit PCA on all hidden states from the full trajectory data
    hidden_all = traj_data['actor_hidden'].reshape(-1, H)
    pca3       = PCA(n_components=3).fit(hidden_all)

    # Select n_traj trajectory starting points (same selection across all n)
    rng     = np.random.default_rng(42)
    N       = len(h_inits)
    sel_idx = rng.choice(N, size=min(n_traj, N), replace=False)
    n_sel   = len(sel_idx)

    # Determine color values: KNN time-to-exit > decoder_dir projection > tab10
    if tte_data is not None:
        color_vals  = patch_progress(h_inits[sel_idx], tte_data, n_neighbors=n_neighbors)
        cbar_label  = f'Patch progress (k={n_neighbors})'
        _cmap       = plt.cm.viridis
    elif decoder_dir is not None:
        color_vals  = h_inits[sel_idx] @ np.array(decoder_dir)
        cbar_label  = f'Initial {proj_label}'
        _cmap       = plt.cm.coolwarm
    else:
        color_vals = None

    if color_vals is not None:
        _norm  = Normalize(vmin=color_vals.min(), vmax=color_vals.max())
        colors = _cmap(_norm(color_vals))
        sm     = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
        sm.set_array([])
    else:
        colors     = plt.cm.tab10(np.arange(n_sel) / 10)
        sm         = None
        cbar_label = None

    def _draw_ax(ax, full_proj, n_a, elev, azim):
        for k in range(n_sel):
            traj_k = full_proj[:, k, :]
            col    = colors[k]
            ax.plot(traj_k[:n_a, 0], traj_k[:n_a, 1], traj_k[:n_a, 2],
                    color=col, linewidth=1.5, linestyle='-', alpha=0.9)
            ax.plot(traj_k[n_a - 1:, 0], traj_k[n_a - 1:, 1], traj_k[n_a - 1:, 2],
                    color=col, linewidth=1.5, linestyle='--', alpha=0.9)
            ax.scatter(*traj_k[0], color=col, s=20, zorder=5)
            ax.scatter(*traj_k[n_a - 1], color=col, s=50, marker='*', zorder=6)
        add_separatrix_plane(ax, params, pca3)
        ax.set_xlabel('PC1', fontsize=8, labelpad=2)
        ax.set_ylabel('PC2', fontsize=8, labelpad=2)
        ax.set_zlabel('PC3', fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

    # Separatrix distance helper
    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    figs = []
    for n in n_vals:
        a_states = trajs_a[:n + 1, sel_idx, :]   # (n+1, n_sel, H)

        trajs_b = np.array(run_states_forward(
            jnp.array(trajs_a[n, sel_idx, :]), input_b, params, network,
            n_steps=n_steps_odor + 1, use_self_action=True))  # (n_steps_odor+1, n_sel, H)

        # Concatenate phases; skip trajs_b[0] as it duplicates a_states[n]
        full_traj = np.concatenate([a_states, trajs_b[1:]], axis=0)  # (n+1+n_steps_odor, n_sel, H)
        T         = full_traj.shape[0]
        full_proj = pca3.transform(full_traj.reshape(-1, H)).reshape(T, n_sel, 3)
        n_a       = n + 1  # number of A-phase points (including initial)

        # Sep distance: (T, n_sel)
        sd = sep_dist(full_traj.reshape(-1, H)).reshape(T, n_sel)

        # KNN time-to-exit along trajectory: (T, n_sel)
        if tte_data is not None:
            knn_traj = patch_progress(
                full_traj.reshape(-1, H), tte_data, n_neighbors=n_neighbors
            ).reshape(T, n_sel)
        else:
            knn_traj = None

        fig = plt.figure(figsize=figsize)
        n_bottom_cols = 2 if knn_traj is not None else 1
        gs  = fig.add_gridspec(2, n_bottom_cols, height_ratios=[5, 1.5], hspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1]) if knn_traj is not None else None

        _draw_ax(ax1, full_proj, n_a, elev1, azim1)
        _draw_ax(ax2, full_proj, n_a, elev2, azim2)

        # Shared time-axis helper
        t_axis = np.arange(T)
        def _plot_time_traces(ax, vals):
            for k in range(n_sel):
                ax.plot(t_axis[:n_a], vals[:n_a, k],
                        color=colors[k], linewidth=1.5, linestyle='-', alpha=0.9)
                ax.plot(t_axis[n_a - 1:], vals[n_a - 1:, k],
                        color=colors[k], linewidth=1.5, linestyle='--', alpha=0.9)
            ax.axvline(n_a - 1, color='k', linewidth=0.8, linestyle=':')

        _plot_time_traces(ax3, sd)
        ax3.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax3.set_xlabel('Time step', fontsize=8)
        ax3.set_ylabel('Sep. distance', fontsize=8)
        format_plot(ax3)

        if ax4 is not None:
            _plot_time_traces(ax4, knn_traj)
            ax4.set_xlabel('Time step', fontsize=8)
            ax4.set_ylabel('Patch progress', fontsize=8)
            format_plot(ax4)

        handles = [
            Line2D([0], [0], color='k', linewidth=1.5, linestyle='-',  label='pre-odor'),
            Line2D([0], [0], color='k', linewidth=1.5, linestyle='--', label='odor'),
        ]
        ax1.legend(handles=handles, fontsize=7, frameon=False, loc='upper left')

        if sm is not None:
            cb = fig.colorbar(sm, ax=[ax1, ax2], pad=0.04, shrink=0.6, label=cbar_label)
            cb.outline.set_visible(False)

        fig.suptitle(f'{n} pre-odor step{"s" if n > 1 else ""}', fontsize=9)
        figs.append(fig)

    return figs


def plot_pre_patch_states_with_next_2d_expect_0(traj_data, params,
                                        figsize=(5, 4), eta=0.1, sep_dist_threshold=0.2,
                                        decoder_dir=None):
    """
    x-axis: pre-odor PC1.  y-axis: sep. distance.
    open circles = initial, solid = after 3 steps (coloured by screen mask).
    States detected via find_pre_odor_onset_states (in_patch + non-NaN ISI).
    Returns (fig, cov_sep_dist_isi) where cov_sep_dist_isi is the scalar
    covariance between post-3-step sep. distance and ISI for screened points.
    """
    events = find_pre_odor_onset_states(traj_data)
    mask   = ~np.isnan(events['isi'])
    if not mask.any():
        print('No valid pre-odor onset events')
        return None, float('nan')

    sub_hidden = events['hidden'][mask, 0]
    sub_next3  = events['hidden'][mask, 3]
    sub_isi    = events['isi'][mask]

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    if decoder_dir is None:
        decoder_dir = fit_remaining_time_decoder(traj_data)
    v  = decoder_dir
    x  = sub_hidden @ v

    def screen_x(x):
        screen = np.zeros_like(x, dtype=bool)
        for i, x0 in enumerate(x):
            next3_within_eta = sub_next3[(x < x0 + eta) & (x > x0 - eta)]
            screen[i] = np.abs(sep_dist(next3_within_eta).mean()) < sep_dist_threshold
        return screen

    screen = screen_x(x)
    d0 = sep_dist(sub_hidden)
    d3 = sep_dist(sub_next3)

    cov_sep_dist_isi = float(np.cov(
        np.stack([d3[screen], sub_isi[screen]], axis=1).T
    )[0, 1])

    norm   = Normalize(vmin=np.nanmin(sub_isi), vmax=np.nanmax(sub_isi))
    cmap   = plt.cm.viridis
    colors = cmap(norm(sub_isi))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, d0, facecolors='none', edgecolors=colors,
               s=10, alpha=1, linewidths=0.8, label='initial')
    ax.scatter(x, d3, c=screen, cmap=cmap, norm=Normalize(vmin=0, vmax=1),
               s=10, alpha=1, linewidths=0, label='after 3 steps')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.02, label='Inter-odor site distance')
    cb.outline.set_visible(False)
    ax.set_xlabel('Remaining time proj.', fontsize=8)
    ax.set_ylabel('Sep. distance', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    format_plot(ax)
    fig.tight_layout()
    return fig, cov_sep_dist_isi

# --- (from cell b7b0aa0e) ---
def run_synthetic_inputs(
    network,
    params,
    input_sequence,
    rng_key=None,
    hidden_size=64,
    n_step_odor_without_stop_thresh=3
):
    """Run the actor RNN on a synthetic sequence of inputs.

    At each timestep the action derived from logits_t (argmax → one-hot) is
    written into indices 4–5 of the next timestep's input, so the network
    receives its own last action as feedback.  The initial last-action is zeros.

    Args:
        network: A2CRNNFlax network instance.
        params: Loaded network parameters.
        input_sequence: Array of shape (T, input_dim), where
            input_dim = obs_size(4) + action_size(2) + prev_reward(1) = 7.
            Indices 4–5 are overridden at every step by the network's last action.
        rng_key: Optional JAX random key for noise. Defaults to a fixed seed.
        hidden_size: Hidden state size (default 64, matching CONFIG).

    Returns:
        actor_hiddens:  Array of shape (T, hidden_size)
        critic_hiddens: Array of shape (T, hidden_size)
        logits:         Array of shape (T, action_size)
        values:         Array of shape (T,)
        actual_inputs:  Array of shape (T, input_dim) — inputs as actually fed to the network
    """
    if rng_key is None:
        rng_key = jax.random.key(0)

    input_sequence = jnp.array(input_sequence)
    n_actions = 2

    actor_hidden  = jnp.zeros((1, hidden_size))
    critic_hidden = jnp.zeros((1, hidden_size))
    last_action   = jnp.zeros(n_actions)   # one-hot of previous action

    all_actor_hiddens  = []
    all_critic_hiddens = []
    all_logits         = []
    all_values         = []
    all_inputs         = []

    for t in range(len(input_sequence)):
        rng_key, noise_key = jax.random.split(rng_key)

        # Override indices 4–5 with last action one-hot
        x_t = input_sequence[t].at[4:6].set(last_action)

        # odor_on_n_timesteps = np.sum([
        #     (all_inputs[-i_n][1:4] > 0.5).any().astype(int) for i_n in range(n_step_odor_without_stop_thresh)
        # ]) 

        # if t > 2 and odor_on_n_timesteps:
        #     x_t = x_t.at[1:4].set(0)

        all_inputs.append(x_t)
        x = x_t[None, :]  # (1, input_dim)

        logits_t, value_t, actor_hidden, critic_hidden, _, _, _ = network.apply(
            params, x, actor_hidden, critic_hidden,
            rngs={'noise': noise_key},
        )

        last_action = jax.nn.one_hot(jnp.argmax(logits_t[0]), n_actions)

        all_actor_hiddens.append(actor_hidden[0])
        all_critic_hiddens.append(critic_hidden[0])
        all_logits.append(logits_t[0])
        all_values.append(value_t[0])

    return (
        jnp.stack(all_actor_hiddens),
        jnp.stack(all_critic_hiddens),
        jnp.stack(all_logits),
        jnp.stack(all_values),
        jnp.stack(all_inputs),
    )


# --- (from cell 26773773) ---
# Input channel labels for the 7-dim input vector
INPUT_LABELS = ['obs_0', 'obs_odor_1', 'obs_odor_2', 'obs_odor_3', 'prev_act_0', 'prev_act_1', 'prev_reward']


def plot_actor_hidden_states(actor_hiddens, input_sequence=None, logits=None, figsize=(12, 6)):
    """Visualise actor hidden-state activity over time.

    Args:
        actor_hiddens:  Array of shape (T, hidden_size) from run_synthetic_inputs.
        input_sequence: Optional array (T, input_dim). Shown as a heatmap above
                        the hidden states so you can relate dynamics to the stimulus.
        logits:         Optional array (T, action_size). Shown as line plots below
                        the hidden-state heatmap.
        figsize: Figure size tuple.
    """
    actor_hiddens = np.array(actor_hiddens)
    T, hidden_size = actor_hiddens.shape

    n_panels = 1 + (input_sequence is not None) + (logits is not None)
    ratios = []
    if input_sequence is not None:
        ratios.append(1)
    ratios.append(3)
    if logits is not None:
        ratios.append(1)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                             gridspec_kw={'height_ratios': ratios})
    if n_panels == 1:
        axes = [axes]
    panel = 0

    # --- optional input heatmap ---
    if input_sequence is not None:
        axes[0].matshow(
            input_sequence.T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
        )
        # plt.grid()
        # add a grid for every square
        axes[0].set_xticks(np.arange(-0.5, input_sequence.shape[0], 1), minor=True)
        axes[0].set_yticks(np.arange(-0.5, input_sequence.shape[1], 1), minor=True)
        axes[0].grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Time step')
        axes[0].set_ylabel('Observation channel')

    # --- hidden-state heatmap ---
    ax_h = axes[1]
    vmax = np.abs(actor_hiddens).max()
    im = ax_h.imshow(actor_hiddens.T, aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest',
                     extent=[0, T, hidden_size - 0.5, -0.5])
    cax = ax_h.inset_axes([1.01, 0, 0.03, 1], transform=ax_h.transAxes)
    plt.colorbar(im, cax=cax, label='Activation')
    ax_h.set_ylabel('Unit index')
    ax_h.set_title('Actor hidden states')
    ax_h.set_xlim(0, T)

    # --- optional logits ---
    if logits is not None:
        logits = np.array(logits)
        ax_l = axes[2]
        for i in range(logits.shape[1]):
            ax_l.plot(np.arange(T), logits[:, i], lw=1, label=f'action {i}')
        ax_l.set_xlim(0, T)
        ax_l.set_ylabel('Logit')
        ax_l.set_xlabel('Timestep')
        ax_l.set_title('Output logits')
        ax_l.legend(fontsize=7)

    plt.tight_layout()


# --- (from cell 11264d6e) ---
def plot_synthetic_in_pc_space(actor_hiddens, pca, pc_activities, color_by=None,
                               color_label='time step', n_pcs=4, figsize=None):
    """Overlay a synthetic hidden-state trajectory on the real-data PC cloud.

    Projects actor_hiddens into the existing pca space (fit on traj_data) and
    plots each adjacent pair of PCs (PC1-2, PC3-4, ...) as a coloured line
    on top of the grey background cloud from pc_activities.

    Args:
        actor_hiddens: Array (T, hidden_size) from run_synthetic_inputs.
        color_by:      Optional 1-D array of length T used to colour the
                       trajectory. Defaults to time step index.
        color_label:   Colorbar label string.
        n_pcs:         Number of PCs to show; must be even. Produces n_pcs/2
                       subplots arranged in a 2-column grid.
        figsize:       Figure size. Auto-sized if None.
    """

    actor_hiddens = np.array(actor_hiddens)
    T = actor_hiddens.shape[0]

    proj = pca.transform(actor_hiddens)  # (T, n_components)

    if color_by is None:
        c_vals = np.arange(T, dtype=float)
    else:
        c_vals = np.array(color_by, dtype=float)

    norm = Normalize(vmin=c_vals.min(), vmax=c_vals.max())
    cmap = plt.cm.plasma

    n_pairs = n_pcs // 2
    n_cols = min(n_pairs, 2)
    n_rows = int(np.ceil(n_pairs / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 4.0 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for pair_idx in range(n_pairs):
        ax = axes_flat[pair_idx]
        pc_x, pc_y = pair_idx * 2, pair_idx * 2 + 1

        # Grey background cloud of real data
        ax.scatter(pc_activities[:, pc_x], pc_activities[:, pc_y],
                   s=0.2, color='#d7d9d7', alpha=0.5, rasterized=True, zorder=0)

        # Synthetic trajectory as coloured line segments
        pts = np.stack([proj[:, pc_x], proj[:, pc_y]], axis=1).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.8,
                            alpha=0.9, zorder=2)
        lc.set_array(c_vals[:-1])
        ax.add_collection(lc)
        ax.autoscale()

        # Mark start and end
        ax.scatter(*proj[0,  [pc_x, pc_y]], s=40, color='green',  zorder=3, label='start')
        ax.scatter(*proj[-1, [pc_x, pc_y]], s=40, color='red',    zorder=3, label='end')

        ax.set_xlabel(f'PC {pc_x + 1}')
        ax.set_ylabel(f'PC {pc_y + 1}')
        ax.set_title(f'PC {pc_x + 1} vs PC {pc_y + 1}')
        if pair_idx == 0:
            ax.legend(fontsize=7, markerscale=0.8)

    # Hide unused axes
    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes_flat[:n_pairs], label=color_label, shrink=0.6)
    plt.tight_layout()
    plt.show()


# --- (from cell 6ad9a180) ---
def plot_patch_segments_with_fixed_points(
    traj_data,
    params,
    network,
    pca,
    trial_idx=0,
    patch_idx=0,
    n_fp_attempts=100,
    fp_tol=1e-6,
    min_segment_len=2,
    threshold=0.5,
    figsize=(8, 22),
    elev=20,
    azim=30,
    elev2=20,
    azim2=120,
    save_dir=None,
    includes_actions_and_rewards=False,
    points_path=None,
    patch_type=None,
):
    """Plot a single patch visit in 3D PC space, split by constant-input segments.

    For each contiguous block where the network input (obs + action + reward) is
    unchanged, shows:
      - Top panel: full patch observation matrix (channels x time) with a red
        rectangle marking the current segment.
      - Bottom panel: 3D PC trajectory for that segment, with fixed points
        found for that input context overlaid as gold stars.

    Args:
        traj_data:       Dict from load/parse, shape (15*20000, ...) per field.
        params:          Network parameters.
        network:         A2CRNNFlax instance.
        trial_idx:       Which of the 15 trials to use (0-14).
        patch_idx:       Which patch within the trial (0-indexed).
        n_fp_attempts:   Random initialisations for fixed-point search.
        fp_tol:          Convergence tolerance for FP optimisation.
        min_segment_len: Skip segments shorter than this many steps.
        threshold:       Obs[0] threshold for patch boundary detection.
        figsize:         Figure size per plot.
    """
    N_STEPS = 20000
    n_total = traj_data['observations'].shape[0]
    n_envs  = n_total // N_STEPS

    n_obs = traj_data['observations'].reshape(n_envs, N_STEPS, -1).shape[-1]

    # Build (n_envs, T, n_obs+2) network-input array: obs + action(1) + reward(1)
    if includes_actions_and_rewards:
        network_inputs = traj_data['observations'].reshape(n_envs, N_STEPS, -1)
    else:
        network_inputs = np.concatenate([
            traj_data['observations'].reshape(n_envs, N_STEPS, -1),
            jax.nn.one_hot(
                traj_data['actions'].reshape(n_envs, N_STEPS, -1).squeeze(axis=-1),
                num_classes=2,
            ),
            traj_data['rewards'].reshape(n_envs, N_STEPS, -1),
        ], axis=-1)

    obs_labels = ['vis.'] + [f'od. {i}' for i in range(1, n_obs - 1)] \
        + (['att.'] if n_obs - int(includes_actions_and_rewards) * 3 == 5 else [])
    INPUT_ROW_LABELS = obs_labels + ['action', 'reward']

    obs    = traj_data['observations'].reshape(n_envs, N_STEPS, -1)[trial_idx]
    hidden = traj_data['actor_hidden'].reshape(n_envs, N_STEPS, -1)[trial_idx]
    inputs = network_inputs[trial_idx]

    # Locate patch boundaries from visual cue (obs[:, 0])
    in_patch = obs[:, 0] > threshold
    padded   = np.concatenate([[False], in_patch, [False]])
    diff     = np.diff(padded.astype(int))
    starts   = np.where(diff ==  1)[0]
    stops    = np.where(diff == -1)[0]

    if patch_type is not None:
        patch_nums = traj_data['current_patch_num'].reshape(n_envs, N_STEPS)[trial_idx]
        mask = np.array([patch_nums[s] == patch_type for s in starts])
        starts = starts[mask]
        stops  = stops[mask]

    if patch_idx >= len(starts):
        label = f'patch_type={patch_type}' if patch_type is not None else f'trial {trial_idx}'
        print(f'Only {len(starts)} patches for {label}; requested patch_idx={patch_idx}')
        return

    p_start, p_stop = starts[patch_idx], stops[patch_idx]
    pre_start = max(0, p_start - 1)
    patch_inputs = inputs[pre_start:p_stop]
    patch_hidden = hidden[pre_start:p_stop]
    T_patch = p_stop - pre_start

    # Split patch into contiguous constant-input segments
    changes   = np.any(patch_inputs[1:] != patch_inputs[:-1], axis=1)
    seg_edges = np.concatenate([[0], np.where(changes)[0] + 1, [T_patch]])
    segments  = [
                    (int(seg_edges[i]), int(seg_edges[i + 1]))
                    for i in range(len(seg_edges) - 1)
                ]

    n_input_dims = patch_inputs.shape[1]

    print(f'Trial {trial_idx}, patch {patch_idx}  '
          f'(global steps {pre_start}–{p_stop - 1}, length {T_patch})  '
          f'→ {len(segments)} segments with ≥{min_segment_len} steps')

    cmap_traj = plt.cm.plasma

    # Load pre-computed points once if a path was provided
    preloaded_fps = None
    if points_path is not None:
        with open(points_path, 'rb') as _f:
            preloaded_fps = jnp.array(pickle.load(_f))

    for seg_idx, (s, e) in enumerate(segments):
        seg_hidden = patch_hidden[s:e+1]
        raw_input  = patch_inputs[s]

        # Build input_vec for FP search: [obs(n_obs), one_hot_action(2), reward(1)]
        if includes_actions_and_rewards:
            obs_end = n_obs - 3
        else:
            obs_end = n_obs
        obs_vec       = raw_input[:obs_end]
        if includes_actions_and_rewards:
            action_idx = int(np.argmax(raw_input[obs_end:obs_end + 2]))
            reward_val = raw_input[obs_end + 2:obs_end + 3]
        else:
            action_idx = int(round(float(raw_input[obs_end])))
            reward_val = raw_input[obs_end + 1:obs_end + 2]
        input_vec     = raw_input

        # Find (or load) fixed points
        if preloaded_fps is not None:
            unique_fps = preloaded_fps
        else:
            key     = jax.random.PRNGKey(42)
            indices = jax.random.randint(key, (n_fp_attempts,), 0,
                                         traj_data['actor_hidden'].shape[0])
            h_inits = jnp.array(traj_data['actor_hidden'][indices])
            fps, converged, _ = find_fixed_points_batch(
                params=params, network=network, input_vec=input_vec,
                h_inits=h_inits, max_steps=60000, learning_rate=0.001,
                tolerance=fp_tol, verbose=False,
            )
            unique_fps = filter_unique_fixed_points(fps, converged)
        n_fps = len(unique_fps)

        # Compute q value (movement speed squared) and participation ratio for fixed points
        fp_pr = float('nan')
        if n_fps > 0:
            h_next = rnn_step_batch(unique_fps, input_vec, params, network)
            fp_q_values = np.array(jnp.sum((unique_fps - h_next) ** 2, axis=-1))
            if n_fps > 1:
                fp_pr = participation_ratio(np.array(unique_fps))

        # --- Figure: observation matrix (top) + two 3D PC plots (middle) + separatrix distance (bottom) ---
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(4, 1, height_ratios=[1, 6, 6, 1], hspace=0.4)
        ax_obs  = fig.add_subplot(gs[0])
        ax_3d   = fig.add_subplot(gs[1], projection='3d')
        ax_3d2  = fig.add_subplot(gs[2], projection='3d')
        ax_sep  = fig.add_subplot(gs[3])

        # Top panel: full patch input matrix, channels x time
        im = ax_obs.imshow(
            patch_inputs.T,
            aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
            interpolation='nearest',
            extent=[-0.5, T_patch - 0.5, n_input_dims - 0.5, -0.5],
        )
        plt.colorbar(im, ax=ax_obs, label='value', fraction=0.03, pad=0.02)
        # ax_obs.set_yticks(range(n_input_dims))
        # ax_obs.set_yticklabels(INPUT_ROW_LABELS, fontsize=7)
        ax_obs.set_xlabel('Time in patch', fontsize=7)
        ax_obs.set_title(
            f'Patch inputs — trial {trial_idx}, patch {patch_idx}  '
            f'(segment {seg_idx + 1}/{len(segments)} highlighted)',
            fontsize=8,
        )
        format_plot(ax_obs)

        # Highlight the current segment with a red rectangle
        rect = Rectangle(
            (s - 0.5, -0.5),          # (x, y) of bottom-left corner
            e - s,                     # width
            n_input_dims,              # height
            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.15,
        )
        ax_obs.add_patch(rect)
        # Red border on top of fill
        rect_border = Rectangle(
            (s - 0.5, -0.5), e - s, n_input_dims,
            linewidth=1.5, edgecolor='red', facecolor='none',
        )
        ax_obs.add_patch(rect_border)

        # Bottom panel: 3D PC trajectory
        patch_proj = pca.transform(patch_hidden)[:, :3]   # full patch, (T_patch, 3)
        proj       = patch_proj[s:e+1]                       # current segment
        T_seg      = len(proj)
        norm       = Normalize(vmin=0, vmax=T_seg - 1)

        # Full patch trajectory coloured by time in patch
        full_pts   = patch_proj.reshape(-1, 1, 3)
        full_segs  = np.concatenate([full_pts[:-1], full_pts[1:]], axis=1)
        t_full     = np.arange(len(patch_proj) - 1, dtype=float)
        full_norm  = Normalize(vmin=0, vmax=len(patch_proj) - 1)
        lc_full    = Line3DCollection(full_segs, cmap=plt.cm.viridis, norm=full_norm,
                                      linewidth=1, alpha=0.4, zorder=1)
        lc_full.set_array(t_full)
        ax_3d.add_collection3d(lc_full)

        # Current segment coloured by side of separatrix
        kernel  = np.array(params['params']['actor']['kernel'])   # (H, 2)
        bias    = np.array(params['params']['actor']['bias'])     # (2,)
        w       = kernel[:, 0] - kernel[:, 1]                     # (H,)
        b_diff  = float(bias[0] - bias[1])
        w_norm  = np.linalg.norm(w)
        scores  = (seg_hidden @ w + b_diff) / w_norm              # (T_seg,)
        abs_max = max(float(np.abs(scores).max()), 1e-8)
        sep_norm = Normalize(vmin=-abs_max, vmax=abs_max)
        cmap_sep = plt.cm.coolwarm

        pts      = proj.reshape(-1, 1, 3)
        seg_segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(seg_segs, cmap=cmap_sep, norm=sep_norm,
                              linewidth=2, alpha=0.9, zorder=2)
        lc.set_array(scores[1:])
        ax_3d.add_collection3d(lc)

        ax_3d.scatter(*proj[0],  s=120, color='green', marker='*', edgecolors='none', zorder=5)
        ax_3d.scatter(*proj[-1], s=120, color='red',   marker='*', edgecolors='none', zorder=5)

        fp_eigenvalues = None
        if n_fps > 0:
            fp_proj = pca.transform(np.array(unique_fps))[:, :3]
            if preloaded_fps is not None:
                ax_3d.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                              s=80, marker='o', color='grey',
                              edgecolors='k', linewidths=0.5, zorder=6)
            else:
                fp_eigenvalues = analyze_stability_batch(unique_fps, input_vec, params, network)
                max_eigs = np.array(jnp.abs(fp_eigenvalues).max(axis=-1))
                eig_vmin = min(max_eigs.min(), 2.0 - max_eigs.max())  # symmetric around 1
                eig_vmax = max(max_eigs.max(), 2.0 - max_eigs.min())
                fp_norm = TwoSlopeNorm(vcenter=1.0,
                                       vmin=min(eig_vmin, 0.999),
                                       vmax=max(eig_vmax, 1.001))
                cmap_fp = plt.cm.coolwarm
                ax_3d.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                              s=80, marker='o', c=max_eigs, cmap=cmap_fp,
                              norm=fp_norm, edgecolors='k', linewidths=0.5, zorder=6)
                sm_fp = plt.cm.ScalarMappable(cmap=cmap_fp, norm=fp_norm)
                cb_fp = plt.colorbar(sm_fp, ax=ax_3d, label='max |eig|',
                                     shrink=0.25, aspect=10, pad=0.12)
                cb_fp.outline.set_visible(False)

        obs_str = ', '.join(f'{v:.2f}' for v in obs_vec)
        ax_3d.set_title(
            f'obs=[{obs_str}]  act={action_idx}  '
            f'rew={float(reward_val[0]):.2f}  FPs={n_fps}  D_PR={fp_pr:.2f}',
            fontsize=8,
        )

        _style_3d_ax(ax_3d, sep_norm, cmap_sep, 'separatrix side', cbar=False)
        ax_3d.view_init(elev=elev, azim=azim)
        add_separatrix_plane(ax_3d, params, pca)

        # Second 3D view
        lc_full2 = Line3DCollection(full_segs, cmap=plt.cm.viridis, norm=full_norm,
                                    linewidth=1, alpha=0.4, zorder=1)
        lc_full2.set_array(t_full)
        ax_3d2.add_collection3d(lc_full2)
        lc2 = Line3DCollection(seg_segs, cmap=cmap_sep, norm=sep_norm,
                               linewidth=2, alpha=0.9, zorder=2)
        lc2.set_array(scores[1:])
        ax_3d2.add_collection3d(lc2)
        ax_3d2.scatter(*proj[0],  s=120, color='green', marker='*', edgecolors='none', zorder=5)
        ax_3d2.scatter(*proj[-1], s=120, color='red',   marker='*', edgecolors='none', zorder=5)
        if n_fps > 0:
            if preloaded_fps is not None:
                ax_3d2.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                               s=80, marker='o', color='grey',
                               edgecolors='k', linewidths=0.5, zorder=6)
            else:
                ax_3d2.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                               s=80, marker='o', c=max_eigs, cmap=cmap_fp,
                               norm=fp_norm, edgecolors='k', linewidths=0.5, zorder=6)
        _style_3d_ax(ax_3d2, sep_norm, cmap_sep, 'separatrix side', cbar=False)
        ax_3d2.view_init(elev=elev2, azim=azim2)
        add_separatrix_plane(ax_3d2, params, pca)

        # Bottom panel: signed distance to separatrix across whole patch
        patch_scores = (patch_hidden @ w + b_diff) / w_norm   # (T_patch,)
        t_patch = np.arange(T_patch)
        ax_sep.axhline(0, color='black', linewidth=1.0, linestyle='--')
        ax_sep.scatter(t_patch, patch_scores,
                       c=patch_scores, cmap='coolwarm',
                       norm=Normalize(vmin=-np.abs(patch_scores).max(),
                                      vmax=np.abs(patch_scores).max()),
                       s=6, linewidths=0)
        ax_sep.axvspan(s, e, color='gray', alpha=0.15)
        ax_sep.set_xlabel('Time in patch', fontsize=7)
        ax_sep.set_ylabel('Separatrix distance', fontsize=7)
        format_plot(ax_sep)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = f'trial{trial_idx}_patch{patch_idx}_seg{seg_idx}.pdf'
            fig.savefig(os.path.join(save_dir, fname), bbox_inches='tight')

        plt.show()

        if fp_eigenvalues is not None:
            plot_eigenspectra_grid(fp_eigenvalues, n_plots=min(n_fps, 9))
            if save_dir is not None:
                fname_spec = f'trial{trial_idx}_patch{patch_idx}_seg{seg_idx}_spectra.pdf'
                plt.savefig(os.path.join(save_dir, fname_spec), bbox_inches='tight')
            plt.show()

def save_fixed_points_for_unique_inputs(
    traj_data,
    params,
    network,
    save_dir,
    network_name,
    pca=None,
    n_fp_attempts=100,
    fp_tol=1e-6,
    min_occurrences=5,
    round_decimals=1,
    includes_actions_and_rewards=False,
    max_fp_steps=60000,
    fp_lr=0.001,
    trial_idx=0,
    patch_idx=0,
    elev=20,
    azim=30,
    figsize=(5, 5),
    threshold=0.5,
    verbose=True,
):
    """Find and save fixed points for all unique input vectors in trajectory data.

    For each unique (rounded) network-input vector found in traj_data, runs
    fixed-point optimisation and saves the resulting (N, H) array as a .pkl
    file under {save_dir}/{network_name}/.  Files are named fp_<input_label>.pkl
    where <input_label> is the rounded input values joined by underscores.

    If pca is provided, also saves a figure per input showing one full patch
    trajectory (trial_idx, patch_idx) coloured by time with fixed points overlaid.

    Args:
        traj_data:                   Dict with 'observations', 'actions', 'reward',
                                     'actor_hidden' arrays (n_total, ...).
        params:                      Network parameters.
        network:                     A2CRNNFlax instance.
        save_dir:                    Root output directory (e.g. '../../../saved_states').
        network_name:                Subdirectory name for this network.
        pca:                         Fitted PCA (optional). When provided, saves a
                                     figure for each input alongside the pkl.
        n_fp_attempts:               Random initialisations per input vector.
        fp_tol:                      Convergence tolerance for FP optimisation.
        min_occurrences:             Skip input vectors seen fewer than this many times.
        round_decimals:              Decimal places for deduplication rounding.
        includes_actions_and_rewards: If True, 'observations' already encodes
                                     action/reward; otherwise they are concatenated.
        max_fp_steps:                Max optimisation steps per input vector.
        fp_lr:                       Adam learning rate for FP search.
        trial_idx:                   Trial to use for the patch trajectory plot.
        patch_idx:                   Which patch within the trial to plot.
        elev, azim:                  3D view angles.
        figsize:                     Figure size for trajectory plots.
        threshold:                   Obs[0] threshold for patch boundary detection.
        verbose:                     Print progress.
    """
    import pickle

    N_STEPS = 20000
    n_total = traj_data['observations'].shape[0]
    n_envs  = n_total // N_STEPS

    # Build flat (n_total, n_input_dims) network-input array
    obs_flat = np.array(traj_data['observations']).reshape(n_total, -1)
    if includes_actions_and_rewards:
        network_inputs = obs_flat
    else:
        actions_flat = np.array(
            jax.nn.one_hot(
                np.array(traj_data['actions']).reshape(n_total).astype(int),
                num_classes=2,
            )
        )
        reward_flat = np.array(traj_data['reward']).reshape(n_total, 1)
        network_inputs = np.concatenate([obs_flat, actions_flat, reward_flat], axis=-1)

    # Identify unique input vectors via rounding
    rounded = np.round(network_inputs, decimals=round_decimals)
    unique_rows, inverse, counts = np.unique(
        rounded, axis=0, return_inverse=True, return_counts=True
    )

    out_dir = os.path.join(save_dir, network_name)
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        n_qualified = int(np.sum(counts >= min_occurrences))
        print(f'{len(unique_rows)} unique input vectors; '
              f'{n_qualified} have >= {min_occurrences} occurrences')

    all_hidden = np.array(traj_data['actor_hidden']).reshape(n_total, -1)
    rng = np.random.default_rng(42)

    # Pre-compute patch hidden states for plotting (done once, reused per input)
    patch_hidden_for_plot = None
    patch_proj_for_plot   = None
    if pca is not None:
        obs_by_env    = np.array(traj_data['observations']).reshape(n_envs, N_STEPS, -1)
        hidden_by_env = all_hidden.reshape(n_envs, N_STEPS, -1)
        obs_trial = obs_by_env[trial_idx]
        in_patch  = obs_trial[:, 0] > threshold
        padded    = np.concatenate([[False], in_patch, [False]])
        diff      = np.diff(padded.astype(int))
        starts    = np.where(diff ==  1)[0]
        stops     = np.where(diff == -1)[0]
        if patch_idx < len(starts):
            p_start = starts[patch_idx]
            p_stop  = stops[patch_idx]
            patch_hidden_for_plot = hidden_by_env[trial_idx, p_start:p_stop]
            patch_proj_for_plot   = pca.transform(patch_hidden_for_plot)[:, :3]
        else:
            if verbose:
                print(f'Warning: trial {trial_idx} has only {len(starts)} patches; '
                      f'skipping plots.')

    for i, (row, count) in enumerate(zip(unique_rows, counts)):
        if count < min_occurrences:
            continue

        label = '_'.join(str(int(round(v))) for v in row)
        out_path = os.path.join(out_dir, f'fp_{label}.pkl')
        input_vec = jnp.array(row)

        init_idxs = rng.integers(0, n_total, n_fp_attempts)
        h_inits = jnp.array(all_hidden[init_idxs])

        if verbose:
            print(f'[{i + 1}/{len(unique_rows)}] {label}  (count={count})')

        fps, converged, _ = find_fixed_points_batch(
            params=params, network=network, input_vec=input_vec,
            h_inits=h_inits, max_steps=max_fp_steps, learning_rate=fp_lr,
            tolerance=fp_tol, verbose=False,
        )
        unique_fps = filter_unique_fixed_points(fps, converged)
        n_fps = len(unique_fps)

        if verbose:
            print(f'  -> {n_fps} unique fixed point(s) saved to {out_path}')

        with open(out_path, 'wb') as f:
            pickle.dump(np.array(unique_fps), f)

        # Plot trajectory + fixed points
        if pca is not None and patch_proj_for_plot is not None:
            T = len(patch_proj_for_plot)
            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(111, projection='3d')

            pts  = patch_proj_for_plot.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(segs, cmap=plt.cm.viridis,
                                    norm=Normalize(vmin=0, vmax=T - 1),
                                    linewidth=1.5, alpha=0.8)
            lc.set_array(np.arange(T - 1, dtype=float))
            ax.add_collection3d(lc)

            if n_fps > 0:
                fp_proj = pca.transform(np.array(unique_fps))[:, :3]
                ax.scatter(fp_proj[:, 0], fp_proj[:, 1], fp_proj[:, 2],
                           s=80, marker='o', color='gold',
                           edgecolors='k', linewidths=0.5, zorder=6)

            ax.set_title(label, fontsize=7)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('PC1', fontsize=7)
            ax.set_ylabel('PC2', fontsize=7)
            ax.set_zlabel('PC3', fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'fp_{label}.pdf'), bbox_inches='tight')
            plt.close(fig)

    if verbose:
        print(f'Done. All results saved under {out_dir}')


def find_exit_site_all_reward_patterns(
    ckpt_path,
    train_state,
    network,
    net_params=None,
    traj_data=None,
    odor_idx=3,
    odor_offset=12,
    odor_on=6,
    odor_off=1,
    n_sites=5,
    input_dim=7,
    hidden_size=64,
    n_stay_threshold=3,
    seed=0,
):
    """Run all 2^n_sites reward patterns and find the first exit site for each.

    Exit criterion: fewer than n_stay_threshold action-0 outputs within a
    single odor-on block.  The run stops as soon as the exit site is found.

    Args:
        ckpt_path:         Path to checkpoint.
        train_state:       Template train state for checkpoint loading.
        network:           A2CRNNFlax instance.
        net_params:        Pre-loaded params (skips loading if provided).
        odor_idx:          Input index for the odor channel (default 3).
        odor_offset:       Timestep of first odor onset (default 12).
        odor_on:           Duration of each odor-on period (default 6).
        odor_off:          Gap between odor-on periods (default 1).
        n_sites:           Number of sites; generates 2^n_sites patterns.
        input_dim:         Input dimensionality (default 7).
        hidden_size:       RNN hidden size (default 64).
        n_stay_threshold:  Exit if stay-action count < this within one block (default 3).

    Returns:
        reward_patterns: bool array (2^n_sites, n_sites); True = rewarded.
        exit_sites:      int array (2^n_sites,); first exit site index, or -1.
        fig:             Matplotlib figure — matrix of reward/exit pattern.
    """
    if net_params is None:
        restored   = checkpoints.restore_checkpoint(
            ckpt_dir=Path(ckpt_path).resolve(), target=train_state)
        net_params = restored.params

    n_patterns      = 2 ** n_sites
    reward_patterns = np.array(
        [[(p >> (n_sites - 1 - i)) & 1 for i in range(n_sites)]
         for p in range(n_patterns)], dtype=bool)
    exit_sites = np.full(n_patterns, -1, dtype=int)

    # Build all input sequences up front: (n_patterns, T_max, input_dim)
    T_max  = n_sites * (odor_on + odor_off) + odor_offset + odor_on + 1
    seq_all = np.zeros((n_patterns, T_max, input_dim))
    seq_all[:, 10:, 0] = 1
    for p_idx, pattern in enumerate(reward_patterns):
        for i in range(n_sites):
            t0 = i * (odor_on + odor_off) + odor_offset
            seq_all[p_idx, t0:t0 + odor_on, odor_idx] = 1
            if pattern[i]:
                seq_all[p_idx, t0 + 4, 6] = 1

    # Batched step: all patterns share the same hidden-state batch
    if traj_data is not None:
        actor_hidden_initial = traj_data['actor_hidden'].reshape(-1, hidden_size)
        critic_hidden_initial = traj_data['critic_hidden'].reshape(-1, hidden_size)
        in_patch_actor = actor_hidden_initial[traj_data['agent_in_patch'].astype(bool)]
        in_patch_critic = critic_hidden_initial[traj_data['agent_in_patch'].astype(bool)]
        rng = np.random.default_rng(seed)
        sample_idx   = rng.integers(len(in_patch_actor))
        actor_hidden  = jnp.tile(in_patch_actor[sample_idx], (n_patterns, 1))
        critic_hidden = jnp.tile(in_patch_critic[sample_idx], (n_patterns, 1))
    else:
        actor_hidden  = jnp.zeros((n_patterns, hidden_size))
        critic_hidden = jnp.zeros((n_patterns, hidden_size))
    last_actions  = jnp.zeros((n_patterns, 2))
    rng_key       = jax.random.key(0)
    done_mask     = np.zeros(n_patterns, dtype=bool)
    t             = 0

    for i in range(n_sites):
        if done_mask.all():
            break
        t0 = i * (odor_on + odor_off) + odor_offset

        # Advance to odor onset (batch)
        while t < t0:
            rng_key, noise_key = jax.random.split(rng_key)
            x_batch = jnp.array(seq_all[:, t, :]).at[:, 4:6].set(last_actions)
            logits_batch, _, actor_hidden, critic_hidden, _, _, _ = network.apply(
                net_params, x_batch, actor_hidden, critic_hidden,
                rngs={'noise': noise_key})
            last_actions = jax.nn.one_hot(jnp.argmax(logits_batch, axis=-1), 2)
            t += 1

        # Run odor-on block (batch); accumulate stay counts
        stay_counts = np.zeros(n_patterns, dtype=int)
        for _ in range(odor_on):
            rng_key, noise_key = jax.random.split(rng_key)
            x_batch = jnp.array(seq_all[:, t, :]).at[:, 4:6].set(last_actions)
            logits_batch, _, actor_hidden, critic_hidden, _, _, _ = network.apply(
                net_params, x_batch, actor_hidden, critic_hidden,
                rngs={'noise': noise_key})
            last_actions  = jax.nn.one_hot(jnp.argmax(logits_batch, axis=-1), 2)
            stay_counts  += (np.array(jnp.argmax(logits_batch, axis=-1)) == 0).astype(int)
            t += 1

        # Mark exit for active patterns that failed the stay threshold
        newly_done = (~done_mask) & (stay_counts < n_stay_threshold)
        exit_sites[newly_done] = i
        done_mask |= newly_done

    # Shuffle before sorting so tie-breaking within reward groups is random
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_patterns)
    reward_patterns = reward_patterns[perm]
    exit_sites      = exit_sites[perm]

    # Precompute cumulative stats shared by both figures
    rp_int = reward_patterns.astype(int)

    rewards_before = np.zeros((n_patterns, n_sites + 1), dtype=int)
    rewards_before[:, 1:] = np.cumsum(rp_int, axis=1)

    trailing_zeros = np.zeros((n_patterns, n_sites + 1), dtype=int)
    for s in range(1, n_sites + 1):
        trailing_zeros[:, s] = np.where(
            rp_int[:, s - 1] == 0,
            trailing_zeros[:, s - 1] + 1,
            0,
        )

    # --- Matrix figure ---
    # 0 = no reward, 1 = reward, 2 = exit site
    matrix = rp_int.copy()  # (n_patterns, n_sites)
    for p_idx, site in enumerate(exit_sites):
        if site >= 0:
            matrix[p_idx, site:] = 2

    # Sort by rewards before exit (ascending); never-exit patterns use total
    # rewards so they mix in naturally; shuffle already randomised ties
    safe_exits = np.maximum(exit_sites, 0)
    rew_key    = np.where(exit_sites >= 0,
                          rewards_before[np.arange(n_patterns), safe_exits],
                          rp_int.sum(axis=1))
    sort_order = np.argsort(rew_key, kind='stable')
    matrix     = matrix[sort_order]

    # Subset to first and last 20 rows, with a separator in between
    n_show = 60
    if n_patterns > 2 * n_show:
        sep_mat = np.full((1, n_sites), -1)
        matrix_display = np.vstack([matrix[:n_show], sep_mat, matrix[-n_show:]])
        rew_col_full    = rew_key[sort_order].astype(float).reshape(-1, 1)
        rew_col_display = np.vstack([rew_col_full[:n_show],
                                     [[np.nan]],
                                     rew_col_full[-n_show:]])
    else:
        matrix_display  = matrix
        rew_col_display = rew_key[sort_order].astype(float).reshape(-1, 1)

    # -1 used as separator value; add white as first colour
    cmap   = plt.matplotlib.colors.ListedColormap(['white', '#d3d3d3', '#4caf50', '#e53935'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm   = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, (ax, ax_rew) = plt.subplots(
        1, 2, figsize=(5, 6.67),
        gridspec_kw={'width_ratios': [n_sites, 3], 'wspace': 0.05},
    )
    ax.imshow(matrix_display, aspect='auto', cmap=cmap, norm=norm,
              interpolation='nearest')
    ax.set_xlabel('Site', fontsize=8)
    ax.set_ylabel('Reward pattern', fontsize=8)
    ax.set_xticks([0, n_sites - 1])
    ax.set_xticklabels([1, n_sites], fontsize=7)
    ax.set_yticks([])

    # Extra column: rewards before exit (or total rewards if no exit)
    rew_col = rew_col_display
    rew_cmap = plt.cm.Blues.copy()
    rew_cmap.set_bad('#e0e0e0')
    im_rew = ax_rew.imshow(
        rew_col, aspect='auto', cmap=rew_cmap,
        vmin=0, vmax=n_sites, interpolation='nearest',
    )
    cb = plt.colorbar(im_rew, ax=ax_rew, label='Rewards before exit',
                      fraction=0.8, pad=0.08, shrink=0.4)
    cb.outline.set_visible(False)
    ax_rew.set_xticks([])
    ax_rew.set_yticks([])

    format_plot(ax)
    format_plot(ax_rew)
    fig.tight_layout()

    # --- Conditional leave-probability matrix ---
    # P[N, M] = P(leave at current site | N total rewards before this site,
    #             M consecutive misses immediately before this site)

    exits_count   = np.zeros((n_sites + 1, n_sites + 1), dtype=int)
    reaches_count = np.zeros((n_sites + 1, n_sites + 1), dtype=int)
    for s in range(n_sites):
        active  = (exit_sites < 0) | (exit_sites >= s)
        N_s     = rewards_before[:, s]
        M_s     = trailing_zeros[:, s]
        np.add.at(reaches_count, (N_s[active],  M_s[active]),  1)
        exiting = exit_sites == s
        np.add.at(exits_count,   (N_s[exiting], M_s[exiting]), 1)

    with np.errstate(invalid='ignore'):
        prob_matrix = np.where(reaches_count > 0,
                               exits_count / reaches_count,
                               np.nan)

    # Trim to rows/cols that have data
    has_row = np.any(reaches_count > 0, axis=1)
    has_col = np.any(reaches_count > 0, axis=0)
    N_max   = int(np.where(has_row)[0].max()) + 1
    M_max   = int(np.where(has_col)[0].max()) + 1
    prob_plot = prob_matrix[:N_max, :M_max]

    fig_prob, ax_prob = plt.subplots(figsize=(4, 2.5))
    im = ax_prob.imshow(
        prob_plot, aspect='auto', cmap='YlOrRd',
        vmin=0, vmax=1, origin='lower', interpolation='nearest',
    )
    plt.colorbar(im, ax=ax_prob, label='P(leave)', fraction=0.046, pad=0.04)
    ax_prob.set_xticks([0, M_max - 1])
    ax_prob.set_xticklabels([0, M_max - 1], fontsize=7)
    ax_prob.set_yticks([0, N_max - 1])
    ax_prob.set_yticklabels([0, N_max - 1], fontsize=7)
    ax_prob.set_xlabel('Consecutive misses', fontsize=8)
    ax_prob.set_ylabel('Total rewards collected', fontsize=8)
    format_plot(ax_prob)
    fig_prob.tight_layout()

    return reward_patterns, exit_sites, fig, fig_prob


def run_specified_reward_patterns_pca(
    reward_patterns_X,
    ckpt_path,
    train_state,
    network,
    net_params=None,
    traj_data=None,
    pca=None,
    odor_idx=3,
    odor_offset=12,
    odor_on=6,
    odor_off=1,
    input_dim=7,
    hidden_size=64,
    n_stay_threshold=3,
    n_initial_states=1,
    seed=0,
    elev=20,
    azim=30,
    figsize=(6, 5),
    tte_data=None,
):
    """Run specified reward patterns and visualise hidden-state trajectories in 3D PCA.

    Args:
        reward_patterns_X: (n_patterns, n_sites) int/bool array. Each row is a
            reward pattern; 1 = rewarded at that site, 0 = not.
        n_initial_states: Number of pre-patch initial states to sample. The same
            states are used across all patterns.
        [remaining args identical to find_exit_site_all_reward_patterns]
        pca: Pre-fit sklearn PCA. Fitted on traj_data if None.

    Returns:
        fig_matrix: Reward-pattern / exit-site matrix figure (n_initial_states rows
                    per pattern block).
        fig_pca:    3-D PCA trajectory figure (one subplot per pattern,
                    n_initial_states trajectories each, coloured by initial state).
        exit_sites: int array (n_initial_states, n_patterns); exit site per
                    (initial_state, pattern), or -1 if no exit.
    """
    reward_patterns_X = np.array(reward_patterns_X, dtype=int)
    n_patterns, n_sites = reward_patterns_X.shape
    batch_size = n_initial_states * n_patterns

    if net_params is None:
        restored   = checkpoints.restore_checkpoint(
            ckpt_dir=Path(ckpt_path).resolve(), target=train_state)
        net_params = restored.params

    # Build input sequences: tile each pattern n_initial_states times
    # batch order: is0_p0, is0_p1, ..., is0_pN, is1_p0, ...
    T_max   = n_sites * (odor_on + odor_off) + odor_offset + odor_on + 1
    seq_pat = np.zeros((n_patterns, T_max, input_dim))
    seq_pat[:, :, 0] = 1
    for p_idx, pattern in enumerate(reward_patterns_X):
        for i in range(n_sites):
            t0 = i * (odor_on + odor_off) + odor_offset
            seq_pat[p_idx, t0:t0 + odor_on, odor_idx] = 1
            if pattern[i]:
                seq_pat[p_idx, t0 + 4, 6] = 1
    seq_batch = np.tile(seq_pat, (n_initial_states, 1, 1))  # (batch_size, T_max, input_dim)

    # Sample n_initial_states pre-patch states (False→True transitions)
    if traj_data is not None:
        in_patch_flat = traj_data['agent_in_patch'].reshape(-1).astype(bool)
        actor_flat    = traj_data['actor_hidden'].reshape(-1, hidden_size)
        critic_flat   = traj_data['critic_hidden'].reshape(-1, hidden_size)
        transitions   = (~in_patch_flat[:-1]) & in_patch_flat[1:]
        pre_patch_idx = np.where(transitions)[0] + 2
        rng     = np.random.default_rng(seed)
        choices = rng.choice(len(pre_patch_idx), size=n_initial_states, replace=False)
        init_actor  = actor_flat[pre_patch_idx[choices]]   # (n_initial_states, H)
        init_critic = critic_flat[pre_patch_idx[choices]]  # (n_initial_states, H)
        # Repeat each initial state n_patterns times
        actor_hidden  = jnp.array(np.repeat(init_actor,  n_patterns, axis=0))
        critic_hidden = jnp.array(np.repeat(init_critic, n_patterns, axis=0))
    else:
        actor_hidden  = jnp.zeros((batch_size, hidden_size))
        critic_hidden = jnp.zeros((batch_size, hidden_size))

    last_actions = jnp.tile(jnp.array([0., 1.]), (batch_size, 1))
    rng_key      = jax.random.key(0)
    done_mask    = np.zeros(batch_size, dtype=bool)

    all_hidden = np.zeros((batch_size, T_max, hidden_size))
    exit_sites_flat = np.full(batch_size, -1, dtype=int)
    exit_times_flat = np.full(batch_size, T_max, dtype=int)
    t = 0

    for i in range(n_sites):
        t0 = i * (odor_on + odor_off) + odor_offset

        while t < t0:
            all_hidden[:, t, :] = np.array(actor_hidden)
            rng_key, noise_key  = jax.random.split(rng_key)
            x_batch = jnp.array(seq_batch[:, t, :]).at[:, 4:6].set(last_actions)
            logits_batch, _, actor_hidden, critic_hidden, _, _, _ = network.apply(
                net_params, x_batch, actor_hidden, critic_hidden,
                rngs={'noise': noise_key})
            last_actions = jax.nn.one_hot(jnp.argmax(logits_batch, axis=-1), 2)
            t += 1

        stay_counts = np.zeros(batch_size, dtype=int)
        for _ in range(odor_on):
            all_hidden[:, t, :] = np.array(actor_hidden)
            rng_key, noise_key  = jax.random.split(rng_key)
            x_batch = jnp.array(seq_batch[:, t, :]).at[:, 4:6].set(last_actions)
            logits_batch, _, actor_hidden, critic_hidden, _, _, _ = network.apply(
                net_params, x_batch, actor_hidden, critic_hidden,
                rngs={'noise': noise_key})
            last_actions    = jax.nn.one_hot(jnp.argmax(logits_batch, axis=-1), 2)
            stay_counts    += (np.array(jnp.argmax(logits_batch, axis=-1)) == 0).astype(int)
            t += 1

        newly_done = (~done_mask) & (stay_counts < n_stay_threshold)
        exit_sites_flat[newly_done] = i
        exit_times_flat[newly_done] = t
        done_mask |= newly_done

    # Reshape to (n_initial_states, n_patterns)
    exit_sites = exit_sites_flat.reshape(n_initial_states, n_patterns)
    exit_times = exit_times_flat.reshape(n_initial_states, n_patterns)

    # Fit PCA if not provided
    if pca is None:
        if traj_data is not None:
            pca = PCA(n_components=3).fit(traj_data['actor_hidden'].reshape(-1, hidden_size))
        else:
            pca = PCA(n_components=3).fit(all_hidden.reshape(-1, hidden_size))

    # --- Matrix figure: n_initial_states rows per pattern block, separated by gaps ---
    cmap_mat = plt.matplotlib.colors.ListedColormap(['white', '#d3d3d3', '#4caf50', '#e53935'])
    bounds   = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm_mat = plt.matplotlib.colors.BoundaryNorm(bounds, cmap_mat.N)

    rows = []
    ytick_pos, ytick_labels = [], []
    row_idx = 0
    for p_idx in range(n_patterns):
        if p_idx > 0:
            rows.append(np.full((1, n_sites), -1))  # separator
            row_idx += 1
        for s_idx in range(n_initial_states):
            row = reward_patterns_X[p_idx].copy()
            site = exit_sites[s_idx, p_idx]
            if site >= 0:
                row[site:] = 2
            rows.append(row[None, :])
            ytick_pos.append(row_idx)
            ytick_labels.append(f'P{p_idx} S{s_idx}')
            row_idx += 1
    matrix_display = np.vstack(rows)

    n_rows_display = matrix_display.shape[0]
    fig_matrix, ax_mat = plt.subplots(
        figsize=(max(3, n_sites * 0.25) * 0.6, (max(2, n_rows_display * 0.3 + 1)) * 0.6))
    ax_mat.imshow(matrix_display, aspect='auto', cmap=cmap_mat, norm=norm_mat,
                  interpolation='nearest')
    ax_mat.set_xlabel('Site', fontsize=8)
    ax_mat.set_ylabel('Seed', fontsize=8)
    _xtick_sites = np.arange(4, n_sites, 5)  # 5, 10, 15, ... (0-indexed positions)
    ax_mat.set_xticks(_xtick_sites)
    ax_mat.set_xticklabels(_xtick_sites + 1, fontsize=7)
    ax_mat.set_yticks(ytick_pos)
    ax_mat.set_yticklabels([], fontsize=7)
    format_plot(ax_mat)
    fig_matrix.tight_layout()

    # --- 3D PCA figure: n_initial_states rows × n_patterns columns ---
    # Each row = one seed; each column = one pattern.
    t_norm = Normalize(vmin=0, vmax=T_max - 1)
    cmap_t = plt.cm.viridis

    fig_pca  = plt.figure(figsize=(figsize[0] * n_patterns, figsize[1] * n_initial_states))
    axes_pca = {}  # keyed (s_idx, p_idx)
    for s_idx in range(n_initial_states):
        for p_idx in range(n_patterns):
            subplot_idx = s_idx * n_patterns + p_idx + 1
            ax3 = fig_pca.add_subplot(n_initial_states, n_patterns, subplot_idx, projection='3d')
            axes_pca[(s_idx, p_idx)] = ax3

    # First pass: draw trajectories, accumulate all projections for unified limits
    all_projs = []
    for s_idx in range(n_initial_states):
        for p_idx in range(n_patterns):
            batch_idx = s_idx * n_patterns + p_idx
            t_end  = exit_times_flat[batch_idx]
            h_traj = all_hidden[batch_idx, :t_end]
            proj   = pca.transform(h_traj)
            all_projs.append(proj)
            ax3 = axes_pca[(s_idx, p_idx)]

            segs   = np.stack([proj[:-1], proj[1:]], axis=1)
            t_vals = np.arange(len(segs), dtype=float)
            lc = Line3DCollection(segs, cmap=cmap_t, norm=t_norm,
                                  linewidth=1.0, alpha=0.9)
            lc.set_array(t_vals)
            ax3.add_collection3d(lc)
            ax3.scatter(*proj[0], color='green', s=20, zorder=5)
            if exit_sites_flat[batch_idx] >= 0:
                ax3.scatter(*proj[-1], color='red', s=20, zorder=5)

            pattern_str = ''.join(str(v) for v in reward_patterns_X[p_idx])
            exit_str    = f'exit@{exit_sites_flat[batch_idx]}' if exit_sites_flat[batch_idx] >= 0 else 'no exit'
            ax3.set_title(f'P{p_idx}:{pattern_str[:6]}.. S{s_idx}\n{exit_str}', fontsize=6)
            ax3.view_init(elev=elev, azim=azim)

    # Unified limits from all projected data
    all_proj_cat = np.concatenate(all_projs, axis=0)
    pad  = 0.05
    xlim = (all_proj_cat[:, 0].min() - pad, all_proj_cat[:, 0].max() + pad)
    ylim = (all_proj_cat[:, 1].min() - pad, all_proj_cat[:, 1].max() + pad)
    zlim = (all_proj_cat[:, 2].min() - pad, all_proj_cat[:, 2].max() + pad)

    ox = xlim[0]; oy = ylim[0]; oz = zlim[0]
    xl = (xlim[1] - xlim[0]) * 0.45
    yl = (ylim[1] - ylim[0]) * 0.45
    zl = (zlim[1] - zlim[0]) * 0.45
    kw = dict(color='k', arrow_length_ratio=0.08, linewidth=0.8)
    for ax3 in axes_pca.values():
        ax3.set_axis_off()
        ax3.set_xlim(xlim); ax3.set_ylim(ylim); ax3.set_zlim(zlim)
        ax3.quiver(ox, oy, oz, xl, 0,  0,  **kw)
        ax3.quiver(ox, oy, oz, 0,  yl, 0,  **kw)
        ax3.quiver(ox, oy, oz, 0,  0,  zl, **kw)
        add_separatrix_plane(ax3, net_params, pca)

    # One colorbar per row
    sm = plt.cm.ScalarMappable(cmap=cmap_t, norm=t_norm)
    sm.set_array([])
    for s_idx in range(n_initial_states):
        row_axes = [axes_pca[(s_idx, p)] for p in range(n_patterns)]
        cb = fig_pca.colorbar(sm, ax=row_axes, pad=0.02,
                              label='Time step', shrink=0.6, aspect=20)
        cb.outline.set_visible(False)

    # Shared pattern labels (used in both bar chart and patch-progress plot)
    all_rewarded   = [np.all(reward_patterns_X[p] == 1) for p in range(n_patterns)]
    all_unrewarded = [np.all(reward_patterns_X[p] == 0) for p in range(n_patterns)]
    def _pattern_label(p):
        if all_rewarded[p]:   return 'Rewarded'
        if all_unrewarded[p]: return 'Unrewarded'
        return ''.join(str(v) for v in reward_patterns_X[p])
    def _pattern_color(p):
        if all_rewarded[p]:   return 'steelblue'
        if all_unrewarded[p]: return 'gray'
        return f'C{p}'

    # --- Patch-progress vs time ---
    fig_pp = None
    if tte_data is not None:
        fig_pp, ax_pp = plt.subplots(figsize=(5, 3))
        for p_idx in range(n_patterns):
            col   = _pattern_color(p_idx)
            label = _pattern_label(p_idx)
            for s_idx in range(n_initial_states):
                batch_idx = s_idx * n_patterns + p_idx
                t_end  = exit_times_flat[batch_idx]
                h_traj = all_hidden[batch_idx, :t_end]
                pp     = patch_progress(h_traj, tte_data)
                ts     = np.arange(len(pp))
                ax_pp.plot(ts, pp, color=col, linewidth=0.8, alpha=0.6,
                           label=label if s_idx == 0 else None)
                ax_pp.scatter(ts[0],  pp[0],  facecolors='none', edgecolors=col,
                              s=20, linewidths=1.0, zorder=5)
                if exit_sites_flat[batch_idx] >= 0:
                    ax_pp.scatter(ts[-1], pp[-1], color=col, s=20, zorder=5)
        ax_pp.set_xlabel('Time step', fontsize=8)
        ax_pp.set_ylabel('Patch progress', fontsize=8)
        handles, labels = ax_pp.get_legend_handles_labels()
        # deduplicate legend entries
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax_pp.legend(seen.values(), seen.keys(), fontsize=7)
        format_plot(ax_pp)
        fig_pp.tight_layout()

    # --- Bar chart: mean sites before opt-out per pattern ---
    # Treat -1 (no exit) as n_sites (agent stayed for all sites)
    sites_before_exit = np.where(exit_sites >= 0, exit_sites, n_sites).astype(float)
    means = sites_before_exit.mean(axis=0)   # (n_patterns,)
    sems  = sites_before_exit.std(axis=0) / np.sqrt(n_initial_states)

    fig_exit_stats, ax_bar = plt.subplots(figsize=(max(3, n_patterns * 1.2), 3))
    x = np.arange(n_patterns)
    ax_bar.bar(x, means, yerr=sems, capsize=4, color='steelblue',
               edgecolor='none', error_kw=dict(linewidth=1))
    tick_labels = [_pattern_label(p) for p in range(n_patterns)]
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tick_labels, fontsize=7)
    ax_bar.set_xlabel('Pattern', fontsize=8)
    ax_bar.set_ylabel('Avg sites before opt-out', fontsize=8)
    ax_bar.set_ylim(0, n_sites + 0.5)
    format_plot(ax_bar)
    fig_exit_stats.tight_layout()

    return fig_matrix, fig_pca, fig_pp, fig_exit_stats, exit_sites


def run_and_plot_synthetic(
    dir_path,
    ckpt_path,
    train_state, 
    network,
    odor_idx=3,
    odor_offset=12,
    odor_on=6,
    odor_off=1,
    n_sites=30,
    T=300,
    reward_pulse=False,
    azim=130,
    elev=80,
    input_dim=7,
    orth_pca=None,
):
    """Load a network, drive it with a synthetic odor sequence, and plot results.

    Loads traj_data and fits PCA, loads checkpoint, builds and runs the input
    sequence, then produces three plots: hidden-state traces, 3-D PC trajectory,
    and FP PC1 projection vs time.
    """

    # Load traj_data and fit PCA
    traj_data = {}
    for env_idx in range(15):
        traj_data_raw = load_trajectory_data(dir_path)
        traj_data_env = parse_behavioral_data(traj_data_raw[env_idx])
        for key in traj_data_env.keys():
            if key in traj_data:
                traj_data[key] = np.concatenate([traj_data[key], traj_data_env[key]], axis=0)
            else:
                traj_data[key] = traj_data_env[key]
    net_pca = PCA()
    net_pca.fit(traj_data['actor_hidden'])

    # Load checkpoint
    restored   = checkpoints.restore_checkpoint(
        ckpt_dir=Path(ckpt_path).resolve(), target=train_state)
    net_params = restored.params
    print('Model loaded.')

    # Build input sequence
    seq = np.zeros((T, input_dim))
    seq[10:, 0] = 1
    for i in range(n_sites):
        t0 = i * (odor_on + odor_off) + odor_offset
        seq[t0:t0 + odor_on, odor_idx] = 1
        if reward_pulse:
            seq[t0 + 4, 6] = 1

    actor_h, critic_h, logits, values, actual_inputs = run_synthetic_inputs(
        network, net_params, jnp.array(seq), rng_key=jax.random.key(0),
    )

    # Hidden-state trace plot
    plot_actor_hidden_states(actor_h, input_sequence=np.array(actual_inputs), logits=logits)

    # 3-D PC trajectory
    traces_3d = np.array(actor_h)[:, None, :]
    plot_trajectory_dyn_3d(traces_3d, net_pca, cbar_label='time step',
                           params=net_params, azim=azim, elev=elev)

    # FP PC1 projection at each odor onset
    _FP_INPUT = jnp.array([1., 0., 0., 0., 0., 1., 0.])
    _key      = jax.random.PRNGKey(42)
    _idx      = jax.random.randint(_key, (200,), 0, traj_data['actor_hidden'].shape[0])
    _h_inits  = jnp.array(traj_data['actor_hidden'][_idx])
    _fps, _conv, _ = find_fixed_points_batch(
        params=net_params, network=network, input_vec=_FP_INPUT,
        h_inits=_h_inits, max_steps=60000, learning_rate=0.001,
        tolerance=1e-3, verbose=False,
    )
    _ufps = np.array(filter_unique_fixed_points(_fps, _conv))
    if len(_ufps) > 0:
        _fp_pca   = PCA(n_components=1).fit(_ufps)
        _pc1_dir  = _fp_pca.components_[0]
        _pc1_mean = _fp_pca.mean_
        _proj = (np.array(actor_h) - _pc1_mean) @ _pc1_dir
        # Find odor onset timesteps from actual_inputs
        _odor_sig = np.array(actual_inputs)[:, odor_idx]
        _onsets   = np.where(np.diff((_odor_sig > 0.5).astype(int)) == 1)[0] + 1
        _proj_at_onsets = _proj[_onsets]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.arange(1, len(_proj_at_onsets) + 1), _proj_at_onsets,
                color='steelblue', lw=1, marker='o', ms=4)
        ax.set_xlabel('Odor cycle')
        ax.set_ylabel('FP PC1 projection')
        format_plot(ax)
        plt.tight_layout()
        plt.show()

        # Orthogonal-to-FP-PC1 trajectories in 2D
        # Segments: onset k → onset k+1 (each starts when odor goes high)
        _h_arr = np.array(actor_h)
        _segs_orth = []
        for k in range(len(_onsets)):
            t_start = _onsets[k]
            t_end   = _onsets[k + 1] if k + 1 < len(_onsets) else len(_h_arr)
            seg = _h_arr[t_start:t_end]
            c    = seg - _pc1_mean
            proj = (c @ _pc1_dir)[:, None] * _pc1_dir[None, :]
            _segs_orth.append(c - proj)

        _all_orth = np.concatenate(_segs_orth, axis=0)
        _orth_pca = orth_pca if orth_pca is not None else PCA(n_components=2).fit(_all_orth)

        fig, ax = plt.subplots(figsize=(5, 5))
        _cmap_cyc = plt.cm.viridis
        _n_segs   = len(_segs_orth)
        for k, seg_orth in enumerate(_segs_orth):
            pts   = _orth_pca.transform(seg_orth)
            color = _cmap_cyc(k / max(_n_segs - 1, 1))
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.8, alpha=0.8)
            ax.scatter(pts[0, 0], pts[0, 1], color=color, s=20, zorder=3)
        sm = plt.cm.ScalarMappable(cmap=_cmap_cyc,
                                    norm=Normalize(vmin=1, vmax=_n_segs))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, label='Odor cycle')
        cb.outline.set_visible(False)
        ax.set_xlabel('Orth PC1')
        ax.set_ylabel('Orth PC2')
        ax.set_title('States orthogonal to FP PC1')
        format_plot(ax)
        plt.tight_layout()
        plt.show()
        return _orth_pca
    return None