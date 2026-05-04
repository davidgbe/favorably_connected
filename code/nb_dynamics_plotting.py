"""Shared plotting and analysis functions for GRU dynamics notebooks."""
import os
from copy import copy
from typing import Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

from aux_funcs import format_plot, format_pc_plot
from network_aux_funcs import compute_separatrix


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
    n_steps=100
):
    """
    Run multiple initial hidden states forward through the RNN.
    
    Args:
        h_inits: Initial hidden states (batch_size, hidden_size)
        input_vec: Fixed input context (input_size,)
        params: Network parameters
        network: A2CRNNFlax instance
        n_steps: Number of forward steps to simulate
    
    Returns:
        trajectories: Array of shape (n_steps, batch_size, hidden_size)
    """
    batch_size = h_inits.shape[0]
    hidden_size = h_inits.shape[1]
    
    # Initialize trajectory storage
    trajectories = jnp.zeros((n_steps, batch_size, hidden_size))
    trajectories = trajectories.at[0].set(h_inits)
    
    # Run forward
    h_current = h_inits
    for t in range(1, n_steps):
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
def extract_patch_starts_and_stops(observations, threshold=0.5):
    """
    Find contiguous periods where observations[..., 0] is above a threshold.

    Parameters
    ----------
    observations : ndarray, shape (T, n_trials, 4)
    threshold : float

    Returns
    -------
    list of n_trials arrays, each shape (n_periods, 2)
        Each row is [start, end) index along the time axis.
    """
    T, n_trials, _ = observations.shape
    results = []

    for trial in range(n_trials):
        signal = observations[:, trial, 0] > threshold  # (T,) bool

        # Pad with False at both ends so diff catches edges
        padded = np.concatenate([[False], signal, [False]])
        diff = np.diff(padded.astype(np.int8))

        starts = np.where(diff == 1)[0]   # rising edges
        ends   = np.where(diff == -1)[0]  # falling edges

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
    obs     = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)   # (T, n_trials, 4)
    hidden  = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)   # (T, n_trials, hidden_size)
    periods = extract_patch_starts_and_stops(obs, threshold=threshold)

    # Pre-compute color signal array if not time-based
    if color_by != 'time':
        raw = traj_data[color_by].reshape(15, 20000, -1).transpose(1, 0, 2)  # (T, n_trials, ...)
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
    plt.show()


# --- (from cell 1f017d61-4055-4a0c-9f44-18729ea9bc9a) ---
def plot_patch_activity_heatmaps(traj_data, threshold=0.5, n_examples=3, figsize=(4, 3)):
    """
    For each of the 3 obs dims (1, 2, 3), plot n_examples patch trajectories as
    heatmaps of raw hidden-unit activity (rows = units, columns = time steps).
    Patches are selected randomly from those where the given dim exceeds threshold.
    """
    obs    = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    periods = extract_patch_starts_and_stops(obs, threshold=threshold)

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


# --- (from cell pre-patch-states-3d) ---
def plot_pre_patch_states_3d(traj_data, pca, threshold=0.5, figsize=(5, 4), color_by='time',
                              show_background=False):
    """
    For each of obs dims 1, 2, 3, scatter the hidden states at the time step
    *just before* that obs dimension crosses above threshold, filtered to
    reward_site_idx > 0 AND agent_in_patch.

    Shows a 2-row × 3-col figure:
      Row 1 – projected with the global PCA (fit on all network states).
      Row 2 – projected with a local PCA fit only on the subselected points.

    Panel title: D_PR all (global) and D_PR sub (subselected points).
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    T, n_trials, H = hidden.shape

    d_pr_all = participation_ratio(traj_data['actor_hidden'])

    if color_by != 'time':
        raw = traj_data[color_by].reshape(15, 20000, -1).transpose(1, 0, 2)
        color_signal = raw[..., 0]
        cbar_label = color_by
    else:
        color_signal = None
        cbar_label = 'time step'

    cmap = plt.cm.viridis
    dims = [(1, 'obs dim 1 onset'), (2, 'obs dim 2 onset'), (3, 'obs dim 3 onset')]

    # Background: subsample all hidden states for the grey cloud
    all_hidden = traj_data['actor_hidden']
    rng_bg  = np.random.default_rng(1)
    n_bg    = min(2000, len(all_hidden))
    bg_idx  = rng_bg.choice(len(all_hidden), size=n_bg, replace=False)
    bg_hidden = all_hidden[bg_idx]
    bg_global = pca.transform(bg_hidden)[:, :3]

    fig = plt.figure(figsize=(figsize[0] * 3, figsize[1] * 2))

    for col, (dim, title) in enumerate(dims):
        sub_hidden = []
        sub_colors = []

        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff   = np.diff(padded.astype(np.int8))
            onsets = np.where(diff == 1)[0]
            for t in onsets:
                pre_t = t - 1
                if pre_t < 0:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])
                if color_signal is not None:
                    if color_by == 'inter_odor_site_distances':
                        sub_colors.append(float(color_signal[t, trial]))
                    else:
                        sub_colors.append(float(color_signal[pre_t, trial]))
                else:
                    sub_colors.append(float(pre_t))

        d_pr_sub = participation_ratio(np.array(sub_hidden)) if len(sub_hidden) > 1 else float('nan')
        panel_title = f'{title}\n$D_{{PR}}$ all={d_pr_all:.1f}  sub={d_pr_sub:.1f}'

        sub_hidden = np.array(sub_hidden)   # (N, H)
        sub_colors = np.array(sub_colors)   # (N,)

        if len(sub_hidden) == 0:
            print(f'No subselected points for dim {dim}')
            continue

        norm = Normalize(vmin=np.nanmin(sub_colors), vmax=np.nanmax(sub_colors))

        # --- Row 1: global PCA ---
        ax_global = fig.add_subplot(2, 3, col + 1, projection='3d')
        ax_global.set_title(panel_title + '\n(global PCA)', fontsize=8)
        if show_background:
            ax_global.scatter(bg_global[:, 0], bg_global[:, 1], bg_global[:, 2],
                              color='lightgray', s=4, alpha=0.3, depthshade=True, zorder=0)
        pts_global = pca.transform(sub_hidden)[:, :3]
        ax_global.scatter(pts_global[:, 0], pts_global[:, 1], pts_global[:, 2],
                          c=sub_colors, cmap=cmap, norm=norm, s=10, alpha=0.6, depthshade=True)
        _style_3d_ax(ax_global, norm, cmap, cbar_label)

        # --- Row 2: local PCA fit on subselected points ---
        ax_local = fig.add_subplot(2, 3, col + 4, projection='3d')
        ax_local.set_title(panel_title + '\n(local PCA)', fontsize=8)
        local_pca = PCA()
        local_pca.fit(sub_hidden)
        pts_local = local_pca.transform(sub_hidden)[:, :3]
        ax_local.scatter(pts_local[:, 0], pts_local[:, 1], pts_local[:, 2],
                         c=sub_colors, cmap=cmap, norm=norm, s=10, alpha=0.6, depthshade=True)
        _style_3d_ax(ax_local, norm, cmap, cbar_label)

    plt.tight_layout()
    plt.show()


# --- (from cell pre-patch-jac-3d) ---
def plot_pre_patch_states_jac_3d(traj_data, params, network, pca, threshold=0.5,
                                  figsize=(5, 4), elev=20, azim=30, downsample=4):
    """
    Like plot_pre_patch_states_3d but colors points by quantities derived from
    the input Jacobian J = dh_next/du (eq. 18 in Driscoll et al.).

    Pre-patch states are selected at odor onset (obs[2] or obs[3] crossing
    threshold). u_ctx encodes the inputs present just before onset (shared
    background context); u_diff encodes only the new input component that
    arrives at onset (the channel going high).  J is evaluated at u_ctx and
    then multiplied by u_diff to isolate the effect of the incoming change.

    Two contexts (rows):
      obs[2] onset  ->  u_ctx = [1,0,0,0,0,1,0],  u_diff = [0,0,1,0,0,0,0]
      obs[3] onset  ->  u_ctx = [1,0,0,0,0,1,0],  u_diff = [0,0,0,1,0,0,0]

    Two scalars (cols):
      col 0: ||J @ u_diff||              (viridis)   — magnitude of state change
      col 1: (J @ u_diff) · w_sep        (coolwarm)  — projection onto separatrix normal
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    T, n_trials, H = hidden.shape

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
        (
            2,
            'obs[2] onset',
            jnp.array([1., 0., 0., 0., 0., 1., 0.]),
            jnp.array([0., 0., 1., 0., 0., 0., 0.])
        ),
        (
            3,
            'obs[3] onset',
            jnp.array([1., 0., 0., 0., 0., 1., 0.]),
            jnp.array([0., 0., 0., 1., 0., 0., 0.])
        ),
    ]

    fig = plt.figure(figsize=(figsize[0] * 2, figsize[1] * 2))

    for row, (dim, ctx_label, u_ctx, u_diff) in enumerate(contexts):
        sub_hidden = []
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))  # compiled per context

        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff   = np.diff(padded.astype(np.int8))
            onsets = np.where(diff == 1)[0]
            for t in onsets:
                pre_t = t - 1
                if pre_t < 0:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])

        if len(sub_hidden) == 0:
            print(f'No points for context {ctx_label}')
            continue

        sub_hidden = np.array(sub_hidden)

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
def plot_pre_patch_states_jac_2d(traj_data, params, network, threshold=0.5,
                                  figsize=(3, 2.2), downsample=4,
                                  n_fp_attempts=200, fp_tol=1e-3,
                                  color_by='reward_site_idx', color_label=None):
    """
    2-D companion to plot_pre_patch_states_jac_3d.

    X-axis: projection of each pre-patch hidden state onto the first PC of the
            fixed points found under u_ctx = [1,0,0,0,0,1,0] (shared by both
            contexts — the background input before odor onset).
    Y-axis: one of the two Jacobian scalars (cols):
      col 0: ||J @ u_diff||              (magnitude of state change)
      col 1: (J @ u_diff) . w_sep        (projection onto separatrix normal)

    Rows correspond to the two odor-onset contexts (obs[2] / obs[3]).
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx   = traj_data['reward_site_idx'].reshape(15, 20000).T
    color_raw  = traj_data[color_by].reshape(15, 20000, -1)[:, :, 0].T  # (T, n_trials)
    print(color_raw)
    _color_label = color_label or color_by
    T, n_trials, H = hidden.shape

    # Separatrix normal
    kernel = np.array(params['params']['actor']['kernel'])  # (H, 2)
    w      = kernel[:, 0] - kernel[:, 1]
    w_hat  = w / np.linalg.norm(w)

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
    _all_h, _all_s = [], []
    for _trial in range(n_trials):
        for _dim in [2, 3]:
            _sig  = obs[:, _trial, _dim] > threshold
            _pad  = np.concatenate([[False], _sig, [False]])
            _ons  = np.where(np.diff(_pad.astype(np.int8)) == 1)[0]
            for _t in _ons:
                _pre = _t - 1
                if _pre < 0:
                    continue
                if not (site_idx[_pre, _trial] > 0 and in_patch[_pre, _trial]):
                    continue
                _all_h.append(hidden[_pre, _trial, :])
                _all_s.append(float(site_idx[_pre, _trial]))
    if len(_all_h) > 1:
        _proj = np.array(_all_h) @ v
        if np.corrcoef(_proj, np.array(_all_s))[0, 1] < 0:
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
        (2, 'obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                             jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        (3, 'obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                             jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[1] * 2))

    for row, (dim, ctx_label, u_ctx, u_diff) in enumerate(contexts):
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))

        sub_hidden  = []
        sub_color   = []
        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff_  = np.diff(padded.astype(np.int8))
            onsets = np.where(diff_ == 1)[0]
            for t in onsets:
                pre_t = t - 1
                if pre_t < 0:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])
                if color_by == 'inter_odor_site_distances':
                    # print(color_raw[pre_t:pre_t +2, trial])
                    sub_color.append(float(color_raw[t, trial]))
                else:
                    sub_color.append(float(color_raw[pre_t, trial]))

        if len(sub_hidden) == 0:
            print(f'No points for context {ctx_label}')
            continue

        sub_hidden = np.array(sub_hidden)
        sub_color  = np.array(sub_color)
        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // downsample), replace=False)
        sub_hidden = sub_hidden[keep]
        sub_color  = sub_color[keep]

        valid      = ~np.isnan(sub_color)
        sub_hidden = sub_hidden[valid]
        sub_color  = sub_color[valid]

        J_batch = np.array(_jac_batch(jnp.array(sub_hidden)))  # (N, H, input_dim)
        Ju      = J_batch @ np.array(u_diff)                   # (N, H)

        scalar_norm = np.linalg.norm(Ju, axis=1)   # (N,)
        scalar_sep  = Ju @ w_hat                   # (N,)
        x_fp        = sub_hidden @ v               # (N,)

        for col, (scalar, ylabel, cmap_name) in enumerate([
            (scalar_norm, r'$\|J \cdot u_{\rm diff}\|$', 'viridis'),
            (scalar_sep,  r'$(Ju)^\top n_{\rm sep}$',      'coolwarm'),
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
            ax.set_xlabel('Projection onto FP PC1', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(ctx_label, fontsize=8)
            format_plot(ax)

    plt.tight_layout()
    return fig


# --- (from cell pre-patch-jac-2d-isi) ---
def plot_pre_patch_states_jac_2d_isi(traj_data, params, network, threshold=0.5,
                                      figsize=(3, 2.2), downsample=4,
                                      n_fp_attempts=200, fp_tol=1e-3):
    """
    Like plot_pre_patch_states_jac_2d but:
      X-axis : inter-odor site distance at the pre-patch state
      Y-axis : |J @ u_diff|  or  (J @ u_diff)^T n_sep
      Color  : projection of the pre-patch state onto FP PC1

    NaN inter-odor site distances are dropped.
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    isi_raw  = traj_data['inter_odor_site_distances'].reshape(15, 20000, -1)[:, :, 0].T
    T, n_trials, H = hidden.shape

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
        (2, 'obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                             jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        (3, 'obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                             jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[1] * 2))

    for row, (dim, ctx_label, u_ctx, u_diff) in enumerate(contexts):
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))

        sub_hidden = []
        sub_isi    = []
        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff_  = np.diff(padded.astype(np.int8))
            onsets = np.where(diff_ == 1)[0]
            for t in onsets:
                pre_t = t - 1
                if pre_t < 0:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])
                sub_isi.append(float(isi_raw[t, trial]))

        if len(sub_hidden) == 0:
            print(f'No points for context {ctx_label}')
            continue

        sub_hidden = np.array(sub_hidden)
        sub_isi    = np.array(sub_isi)
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


def plot_pre_patch_states_jac_pc1_space(traj_data, params, network, threshold=0.5,
                                         figsize=(3, 2.2), downsample=4,
                                         n_fp_attempts=200, fp_tol=1e-3):
    """
    2×2 grid — rows: obs[2] / obs[3] onset contexts; cols: two Jacobian scalars.

    X-axis : projection of the pre-patch hidden state onto FP PC1
             (centred at the fixed-point mean).
    Y-axis : orthogonal distance of the hidden state from the FP PC1 axis.
    Color  : col 0 — ||J·u_diff||   (viridis)
             col 1 — (J·u_diff)·n_sep  (coolwarm, symmetric)
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    T, n_trials, H = hidden.shape

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
        (2, 'obs[2] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                             jnp.array([0., 0., 1., 0., 0., 0., 0.])),
        (3, 'obs[3] onset', jnp.array([1., 0., 0., 0., 0., 1., 0.]),
                             jnp.array([0., 0., 0., 1., 0., 0., 0.])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(figsize[0] * 2, figsize[1] * 2))

    for row, (dim, ctx_label, u_ctx, u_diff) in enumerate(contexts):
        _jac_batch = jax.jit(jax.vmap(lambda h: _jac_fn(h, u_ctx)))

        sub_hidden = []
        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff_  = np.diff(padded.astype(np.int8))
            onsets = np.where(diff_ == 1)[0]
            for t in onsets:
                pre_t = t - 1
                if pre_t < 0:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])

        if len(sub_hidden) == 0:
            print(f'No points for context {ctx_label}')
            continue

        sub_hidden = np.array(sub_hidden)
        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // downsample),
                          replace=False)
        sub_hidden = sub_hidden[keep]

        J_batch     = np.array(_jac_batch(jnp.array(sub_hidden)))  # (N, H, input_dim)
        Ju          = J_batch @ np.array(u_diff)                   # (N, H)
        scalar_norm = np.linalg.norm(Ju, axis=1)                   # (N,)
        scalar_sep  = Ju @ w_hat                                   # (N,)

        x_fp   = (sub_hidden - fp_mean) @ v   # PC1 projection (centred)
        y_orth = _orth_dist_fp(sub_hidden)     # orthogonal distance

        for col, (scalar, clabel, cmap_name) in enumerate([
            (scalar_norm, r'$\|J \cdot u_{\rm diff}\|$',  'viridis'),
            (scalar_sep,  r'$(Ju)^\top n_{\rm sep}$',      'coolwarm'),
        ]):
            ax = axes[row, col]
            cmap  = plt.get_cmap(cmap_name)
            cnorm = Normalize(vmin=scalar.min(), vmax=scalar.max())

            sc = ax.scatter(x_fp, y_orth, c=scalar, cmap=cmap, norm=cnorm,
                            s=6, alpha=0.8, linewidths=0)
            cb = plt.colorbar(sc, ax=ax, pad=0.02, label=clabel)
            cb.solids.set_alpha(1)
            cb.outline.set_visible(False)
            ax.set_xlabel('FP PC1 projection', fontsize=8)
            ax.set_ylabel('Distance from FP PC1 axis', fontsize=8)
            ax.set_title(ctx_label, fontsize=8)
            format_plot(ax)

    plt.tight_layout()
    plt.show()


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

    hidden  = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)  # (T, n_trials, H)
    rewards = traj_data['rewards_seen_in_patch'].reshape(15, 20000, -1).transpose(1, 0, 2)  # (T, n_trials, ...)

    H = hidden.shape[2]
    reward_flat = rewards[..., 0].ravel()   # (T*n_trials,)
    hidden_flat = hidden.reshape(-1, H)     # (T*n_trials, H)

    rs = np.array([pearsonr(hidden_flat[:, h], reward_flat)[0] for h in range(H)])

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(rs, bins=30, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Pearson r (neuron activity vs rewards_seen_in_patch)')
    ax.set_ylabel('Neuron count')
    ax.set_title('Neuron–reward correlation distribution')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()

    # --- Scatter for top_n neurons ---
    top_idx = np.argsort(np.abs(rs))[::-1][:top_n]
    ncols = min(top_n, 5)
    fig, axes = plt.subplots(1, ncols, figsize=(figsize[0] * ncols / 2, figsize[1]), sharey=False)
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
    plt.suptitle(f'Top {top_n} neurons by |r|', y=1.02)
    plt.tight_layout()
    plt.show()

    return rs


# --- (from cell pre-patch-with-next) ---
def plot_pre_patch_states_with_next_3d(traj_data, params, pca, threshold=0.5,
                                        figsize=(5, 4), color_by='time', cbar_label='Time',
                                        elev=20, azim=30):
    """
    Two figures are produced (obs dim 2 and obs dim 3 contexts only):

    Figure 1 — two 3-D panels: pre-patch hidden state (scatter) with a line
               through the next three timesteps, plus the separatrix plane.

    Figure 2 — two 2-D panels: initial signed separatrix distance (steelblue)
               and distance after 3 steps (tomato) vs the first PC of the
               initial states for that context.
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    T, n_trials, H = hidden.shape

    d_pr_all = participation_ratio(traj_data['actor_hidden'])

    if color_by != 'time':
        raw = traj_data[color_by].reshape(15, 20000, -1).transpose(1, 0, 2)
        color_signal = raw[..., 0]
    else:
        color_signal = None

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    cmap = plt.cm.viridis
    dims = [(2, 'obs dim 2 onset'), (3, 'obs dim 3 onset')]

    fig1, axes_3d = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]),
                                  subplot_kw={'projection': '3d'})
    fig2, axes_2d = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    for col, (dim, title) in enumerate(dims):
        sub_hidden = []
        sub_next1  = []
        sub_next2  = []
        sub_next3  = []
        sub_colors = []

        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff   = np.diff(padded.astype(np.int8))
            onsets = np.where(diff == 1)[0]
            for t in onsets:
                pre_t = t - 1
                n1_t  = t
                n2_t  = t + 1
                n3_t  = t + 2
                if pre_t < 0 or n3_t >= T:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])
                sub_next1.append( hidden[n1_t, trial, :])
                sub_next2.append( hidden[n2_t, trial, :])
                sub_next3.append( hidden[n3_t, trial, :])
                if color_signal is not None:
                    if color_by == 'inter_odor_site_distances':
                        sub_colors.append(float(color_signal[t, trial]))
                    else:
                        sub_colors.append(float(color_signal[pre_t, trial]))
                else:
                    sub_colors.append(float(pre_t))

        if len(sub_hidden) == 0:
            print(f'No subselected points for dim {dim}')
            continue

        sub_hidden = np.array(sub_hidden)
        sub_next1  = np.array(sub_next1)
        sub_next2  = np.array(sub_next2)
        sub_next3  = np.array(sub_next3)
        sub_colors = np.array(sub_colors)

        # Save full arrays for 2D plot before subsampling for 3D
        full_hidden = sub_hidden.copy()
        full_next3  = sub_next3.copy()

        rng  = np.random.default_rng(0)
        keep = rng.choice(len(sub_hidden), size=max(1, len(sub_hidden) // 40), replace=False)
        sub_hidden = sub_hidden[keep]
        sub_next1  = sub_next1[keep]
        sub_next2  = sub_next2[keep]
        sub_next3  = sub_next3[keep]
        sub_colors = sub_colors[keep]

        d_pr_sub    = participation_ratio(sub_hidden) if len(sub_hidden) > 1 else float('nan')
        panel_title = f'{title}\n$D_{{PR}}$ all={d_pr_all:.1f}  sub={d_pr_sub:.1f}'
        norm = Normalize(vmin=np.nanmin(sub_colors), vmax=np.nanmax(sub_colors))

        # --- Figure 1: 3-D panel ---
        def _add_lines(ax, p0, p1, p2, p3, colors):
            segs = np.stack([p0, p1, p2, p3], axis=1)  # (N, 4, 3)
            lc = Line3DCollection(segs, colors=cmap(norm(colors)),
                                  linewidths=0.6, alpha=0.5, zorder=1)
            ax.add_collection3d(lc)

        ax3 = axes_3d[col]
        ax3.set_title(panel_title, fontsize=8)
        pts_pre = pca.transform(sub_hidden)[:, :3]
        pts_n1  = pca.transform(sub_next1)[:, :3]
        pts_n2  = pca.transform(sub_next2)[:, :3]
        pts_n3  = pca.transform(sub_next3)[:, :3]
        _add_lines(ax3, pts_pre, pts_n1, pts_n2, pts_n3, sub_colors)
        ax3.scatter(pts_pre[:, 0], pts_pre[:, 1], pts_pre[:, 2],
                    c=sub_colors, cmap=cmap, norm=norm,
                    s=12, alpha=0.6, depthshade=True)
        _style_3d_ax(ax3, norm, cmap, cbar_label)
        add_separatrix_plane(ax3, params, pca)
        ax3.view_init(elev=elev, azim=azim)

        # --- Figure 2: 2-D panel (all points, no subsampling) ---
        ctx_pca = PCA(n_components=1)
        ctx_pca.fit(full_hidden)
        v  = -ctx_pca.components_[0]
        x  = full_hidden @ v
        d0 = sep_dist(full_hidden)
        d3 = sep_dist(full_next3)

        ax2 = axes_2d[col]
        ax2.scatter(x, d0, color='steelblue', s=2, alpha=1,
                    linewidths=0, label='initial')
        ax2.scatter(x, d3, color='tomato',    s=2, alpha=1,
                    linewidths=0, label='after 3 steps')
        ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Projection onto pre-odor PC1', fontsize=8)
        ax2.set_ylabel('Sep. distance', fontsize=8)
        ax2.set_title(title, fontsize=8)
        ax2.legend(fontsize=7, frameon=False)
        format_plot(ax2)

    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2


def plot_pre_patch_states_with_next_2d(traj_data, params, pca, threshold=0.5,
                                        figsize=(5, 4)):
    """
    2-D-only companion to plot_pre_patch_states_with_next_3d.

    One figure, two panels (obs dim 2 / dim 3 onset contexts):
      open circles  — pre-patch (initial) hidden state
      solid circles — state after 3 steps
    Both are coloured by inter-odor site distance (viridis). NaN ISI dropped.
    x-axis: projection onto context PC1 of initial states.
    y-axis: signed separatrix distance.
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    isi_raw  = traj_data['inter_odor_site_distances'].reshape(15, 20000, -1)[:, :, 0].T
    T, n_trials, H = hidden.shape

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    dims = [(2, 'obs dim 2 onset'), (3, 'obs dim 3 onset')]
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    for col, (dim, title) in enumerate(dims):
        sub_hidden = []
        sub_next3  = []
        sub_isi    = []

        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff   = np.diff(padded.astype(np.int8))
            onsets = np.where(diff == 1)[0]
            for t in onsets:
                pre_t = t - 1
                n3_t  = t + 2
                if pre_t < 0 or n3_t >= T:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])
                sub_next3.append(hidden[n3_t,  trial, :])
                sub_isi.append(float(isi_raw[t, trial]))

        if len(sub_hidden) == 0:
            print(f'No points for dim {dim}')
            continue

        sub_hidden = np.array(sub_hidden)
        sub_next3  = np.array(sub_next3)
        sub_isi    = np.array(sub_isi)

        valid      = ~np.isnan(sub_isi)
        sub_hidden = sub_hidden[valid]
        sub_next3  = sub_next3[valid]
        sub_isi    = sub_isi[valid]

        if len(sub_hidden) == 0:
            print(f'All NaN ISI for dim {dim}')
            continue

        ctx_pca = PCA(n_components=1)
        ctx_pca.fit(sub_hidden)
        v  = -ctx_pca.components_[0]
        x  = sub_hidden @ v
        d0 = sep_dist(sub_hidden)
        d3 = sep_dist(sub_next3)

        norm   = Normalize(vmin=np.nanmin(sub_isi), vmax=np.nanmax(sub_isi))
        cmap   = plt.cm.viridis
        colors = cmap(norm(sub_isi))

        ax = axes[col]
        ax.scatter(x, d0, facecolors='none', edgecolors=colors,
                   s=24, alpha=0.6, linewidths=0.8, label='initial')
        ax.scatter(x, d3, c=sub_isi, cmap=cmap, norm=norm,
                   s=24, alpha=0.6, linewidths=0, label='after 3 steps')
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, label='Inter-odor site distance')
        cb.outline.set_visible(False)

        ax.set_xlabel('Projection onto pre-odor PC1', fontsize=8)
        ax.set_ylabel('Sep. distance', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        format_plot(ax)

    fig.tight_layout()
    return fig


def plot_pre_patch_states_fp_pc1_2d(traj_data, params, network, threshold=0.5,
                                     figsize=(5, 4), n_fp_attempts=200, fp_tol=1e-3):
    """
    Like plot_pre_patch_states_with_next_2d but x-axis is projection onto the
    first PC of stable fixed points found for input [1, 0, 0, 0, 0, 1, 0].

    One figure, two panels (obs dim 2 / dim 3 onset contexts):
      open circles  — pre-patch (initial) hidden state
      solid circles — state after 3 steps
    Both coloured by reward site index (viridis).
    x-axis: projection onto FP PC1 (centred at FP mean).
    y-axis: signed separatrix distance.
    """
    obs      = traj_data['observations'].reshape(15, 20000, -1).transpose(1, 0, 2)
    hidden   = traj_data['actor_hidden'].reshape(15, 20000, -1).transpose(1, 0, 2)
    in_patch = traj_data['agent_in_patch'].reshape(15, 20000).T.astype(bool)
    site_idx = traj_data['reward_site_idx'].reshape(15, 20000).T
    T, n_trials, H = hidden.shape

    kernel = np.array(params['params']['actor']['kernel'])
    w      = kernel[:, 0] - kernel[:, 1]
    b_diff = float(np.array(params['params']['actor']['bias'])[0] -
                   np.array(params['params']['actor']['bias'])[1])
    w_norm = np.linalg.norm(w)
    def sep_dist(h): return (h @ w + b_diff) / w_norm

    # Find stable fixed points for input [1, 0, 0, 0, 0, 1, 0]
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
    v       = -fp_pca.components_[0]  # (H,) — negated for consistent sign convention
    fp_mean = fp_pca.mean_             # (H,)

    dims = [(2, 'obs dim 2 onset'), (3, 'obs dim 3 onset')]
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    for col, (dim, title) in enumerate(dims):
        sub_hidden = []
        sub_next3  = []
        sub_site   = []

        for trial in range(n_trials):
            signal = obs[:, trial, dim] > threshold
            padded = np.concatenate([[False], signal, [False]])
            diff   = np.diff(padded.astype(np.int8))
            onsets = np.where(diff == 1)[0]
            for t in onsets:
                pre_t = t - 1
                n3_t  = t + 2
                if pre_t < 0 or n3_t >= T:
                    continue
                if not (site_idx[pre_t, trial] > 0 and in_patch[pre_t, trial]):
                    continue
                sub_hidden.append(hidden[pre_t, trial, :])
                sub_next3.append(hidden[n3_t,  trial, :])
                sub_site.append(float(site_idx[pre_t, trial]))

        if len(sub_hidden) == 0:
            print(f'No points for dim {dim}')
            continue

        sub_hidden = np.array(sub_hidden)
        sub_next3  = np.array(sub_next3)
        sub_site   = np.array(sub_site)

        x0 = (sub_hidden - fp_mean) @ v
        x3 = (sub_next3  - fp_mean) @ v
        d0 = sep_dist(sub_hidden)
        d3 = sep_dist(sub_next3)

        norm   = Normalize(vmin=sub_site.min(), vmax=sub_site.max())
        cmap   = plt.cm.viridis
        colors = cmap(norm(sub_site))

        ax = axes[col]
        ax.scatter(x0, d0, facecolors='none', edgecolors=colors,
                   s=24, alpha=0.6, linewidths=0.8, label='initial')
        ax.scatter(x3, d3, c=sub_site, cmap=cmap, norm=norm,
                   s=24, alpha=0.6, linewidths=0, label='after 3 steps')
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02, label='Reward site index')
        cb.outline.set_visible(False)

        ax.set_xlabel('Projection onto FP PC1', fontsize=8)
        ax.set_ylabel('Sep. distance', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        format_plot(ax)

    fig.tight_layout()
    return fig


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
    figsize=(7, 7),
    elev=20,
    azim=30,
    save_dir=None,
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

    # Build (15, T, 6) network-input array: obs(4) + action(1) + reward(1)
    network_inputs = np.concatenate([
        traj_data['observations'].reshape(15, N_STEPS, -1),
        traj_data['actions'].reshape(15, N_STEPS, -1),
        traj_data['reward'].reshape(15, N_STEPS, -1),
    ], axis=-1)

    INPUT_ROW_LABELS = ['vis.', 'od. 1', 'od. 2', 'od. 3', 'action', 'reward']

    obs    = traj_data['observations'].reshape(15, N_STEPS, -1)[trial_idx]  # (T, 4)
    hidden = traj_data['actor_hidden'].reshape(15, N_STEPS, -1)[trial_idx]  # (T, H)
    inputs = network_inputs[trial_idx]                                         # (T, 6)

    # Locate patch boundaries from visual cue (obs[:, 0])
    in_patch = obs[:, 0] > threshold
    padded   = np.concatenate([[False], in_patch, [False]])
    diff     = np.diff(padded.astype(int))
    starts   = np.where(diff ==  1)[0]
    stops    = np.where(diff == -1)[0]

    if patch_idx >= len(starts):
        print(f'Only {len(starts)} patches in trial {trial_idx}; '
              f'requested patch_idx={patch_idx}')
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

    for seg_idx, (s, e) in enumerate(segments):
        s_eff = 0 if seg_idx == 0 else s-1
        seg_hidden = patch_hidden[s_eff:e]
        raw_input  = patch_inputs[s]

        # Build 7-dim input_vec for FP search: [obs(4), one_hot_action(2), reward(1)]
        obs_vec       = raw_input[:4]
        action_idx    = int(round(float(raw_input[4])))
        reward_val    = raw_input[5:6]
        action_onehot = jax.nn.one_hot(action_idx, num_classes=2)
        input_vec     = jnp.concatenate([jnp.array(obs_vec), action_onehot,
                                         jnp.array(reward_val)])

        # Find fixed points
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

        # --- Figure: observation matrix (top) + 3D PC plot (middle) + separatrix distance (bottom) ---
        fig = plt.figure(figsize=figsize)
        gs  = fig.add_gridspec(3, 1, height_ratios=[1, 3, 1], hspace=0.55)
        ax_obs  = fig.add_subplot(gs[0])
        ax_3d   = fig.add_subplot(gs[1], projection='3d')
        ax_sep  = fig.add_subplot(gs[2])

        # Top panel: full patch input matrix, channels x time
        im = ax_obs.imshow(
            patch_inputs.T,
            aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
            interpolation='nearest',
            extent=[-0.5, T_patch - 0.5, n_input_dims - 0.5, -0.5],
        )
        plt.colorbar(im, ax=ax_obs, label='value', fraction=0.03, pad=0.02)
        ax_obs.set_yticks(range(n_input_dims))
        ax_obs.set_yticklabels(INPUT_ROW_LABELS, fontsize=7)
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
        proj       = patch_proj[s_eff:e]                       # current segment
        T_seg      = len(proj)
        norm       = Normalize(vmin=0, vmax=T_seg - 1)

        # Full patch trajectory at low alpha for context
        full_pts  = patch_proj.reshape(-1, 1, 3)
        full_segs = np.concatenate([full_pts[:-1], full_pts[1:]], axis=1)
        lc_full   = Line3DCollection(full_segs, color='#aaaaaa', linewidth=1,
                                     alpha=0.25, zorder=1)
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
