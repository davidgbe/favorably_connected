from flax import linen as nn
import jax.numpy as jnp
from jax import random
from agents.networks.vanilla_rnn_cell_flax import VanillaRNNCell

class A2CRNNFlax(nn.Module):
    """Flax implementation of A2C RNN network"""
    action_size: int
    hidden_size: int
    rnn_type: str = 'GRU'  # 'VANILLA' or 'GRU'
    var_noise: float = 1e-4
    
    def setup(self):
        if self.rnn_type == 'VANILLA':
            self.rnn_actor = VanillaRNNCell(self.hidden_size, self.var_noise)
            self.rnn_critic = VanillaRNNCell(self.hidden_size, self.var_noise)
        elif self.rnn_type == 'GRU':
            self.rnn_actor = nn.GRUCell(
                features=self.hidden_size,
                kernel_init=nn.initializers.orthogonal(),
            )
            self.rnn_critic = nn.GRUCell(
                features=self.hidden_size,
                kernel_init=nn.initializers.orthogonal(),
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")
            
        # Actor and critic heads
        self.actor = nn.Dense(self.action_size)
        self.critic = nn.Dense(1)
        self.env_quality_prediction = nn.Dense(3) # for 3 environment parameters

        self.actor_value_heads = nn.Dense(3)
        self.actor_value_deriv_heads = nn.Dense(3)
        
    def __call__(self, x, actor_hidden, critic_hidden):
        """
        Args:
            x: input (batch_size, input_dim)
            actor_hidden: (batch_size, hidden_size) 
            critic_hidden: (batch_size, hidden_size)
        Returns:
            logits, value, new_actor_hidden, new_critic_hidden
        """
        # Actor RNN
        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x)
        logits = self.actor(actor_outputs)
        
        # Critic RNN  
        new_critic_hidden, critic_outputs = self.rnn_critic(critic_hidden, x)
        value = self.critic(critic_outputs).squeeze(-1)

        pred_env_quality = self.env_quality_prediction(actor_outputs)

        actor_values = self.actor_value_heads(actor_outputs)
        actor_value_derivs = self.actor_value_deriv_heads(actor_outputs)
        
        return logits, value, new_actor_hidden, new_critic_hidden, pred_env_quality, actor_values
    
    
def init_network_and_params(
    hidden_size: int,
    action_size: int,
    obs_size: int,
    rnn_type: str,
    var_noise: float,
    rng_key: jnp.ndarray,
):
    # Network input size: obs + prev_obs + prev_action + prev_reward
    input_size = obs_size + action_size + 1

    # Initialize network
    network = A2CRNNFlax(
        action_size=action_size,
        hidden_size=hidden_size, 
        rnn_type=rnn_type,
        var_noise=var_noise,
    )
    
    # Initialize parameters
    param_key, hidden_key = random.split(rng_key, 2)
    dummy_input = jnp.zeros((1, input_size))
    dummy_hidden = jnp.zeros((1, hidden_size))
    
    params = network.init(
        param_key, 
        dummy_input, 
        dummy_hidden, 
        dummy_hidden,
    )

    return network, params