from flax import linen as nn
from jax import random
import jax.numpy as jnp

class VanillaRNNCell(nn.Module):
    """Vanilla RNN cell with custom initializers"""
    hidden_size: int
    var_noise: float = 0
    
    def setup(self):
        # Input projection with default initializers
        self.input_projection = nn.Dense(
            self.hidden_size, 
            use_bias=False,
        )
        
        # Hidden projection with custom initializers
        self.hidden_projection = nn.Dense(
            self.hidden_size, 
            use_bias=True,
            kernel_init=nn.initializers.orthogonal(),  # Orthogonal weights
            bias_init=nn.initializers.zeros,           # Zero bias
        )
        
    def __call__(self, hidden, x):
        """
        Args:
            x: input (batch_size, input_dim)
            hidden: previous hidden state (batch_size, hidden_size)
        Returns:
            new_hidden: (batch_size, hidden_size)
        """
        input_contrib = self.input_projection(x)
        new_hidden = nn.relu(self.hidden_projection(hidden + input_contrib))
        
        # Add noise if specified
        noise = random.normal(
            self.make_rng('noise'), 
            new_hidden.shape
        ) * jnp.sqrt(self.var_noise)
        new_hidden = new_hidden + noise
            
        return new_hidden, new_hidden