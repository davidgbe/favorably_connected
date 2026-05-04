import jax.numpy as jnp

def compute_separatrix(weights, bias, l=100):
    """
    Compute the separatrix for a linear readout.
    """

    y_star = jnp.array([[0., 0.], [l, -l]])
    return jnp.linalg.pinv(weights) @ (y_star - bias)