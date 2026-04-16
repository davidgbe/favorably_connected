from flax import linen as nn
import jax.numpy as jnp
from jax import random
from agents.networks.vanilla_rnn_cell_flax import VanillaRNNCell

class A2CRNNFlax(nn.Module):
    """Flax implementation of A2C RNN network"""
    action_size: int
    obs_size: int
    hidden_size: int
    rnn_type: str = 'GRU'  # 'VANILLA' or 'GRU'
    unit_noise_std: float = 1e-2
    
    def setup(self):
        if self.rnn_type == 'VANILLA':
            self.rnn_actor = VanillaRNNCell(self.hidden_size, self.unit_noise_std)
            self.rnn_critic = VanillaRNNCell(self.hidden_size, self.unit_noise_std)
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
        self.exp_filtered_reward_rate_prediction = nn.Dense(1) # for 3 environment parameters

        self.obs_pred_hidden_size = 16
        self.obs_pred_layer_1 = nn.Dense(self.obs_pred_hidden_size)
        self.obs_prediction = nn.Dense(self.obs_size + 1)
        
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

        # Critic RNN
        new_critic_hidden, critic_outputs = self.rnn_critic(critic_hidden, x)

        # Apply noise to hidden states
        noise_actor = random.normal(
            self.make_rng('noise'),
            new_actor_hidden.shape
        ) * self.unit_noise_std
        new_actor_hidden = new_actor_hidden + noise_actor

        noise_critic = random.normal(
            self.make_rng('noise'),
            new_critic_hidden.shape
        ) * self.unit_noise_std
        new_critic_hidden = new_critic_hidden + noise_critic

        logits = self.actor(actor_outputs)
        value = self.critic(critic_outputs).squeeze(-1)

        pred_env_quality = self.env_quality_prediction(actor_outputs)
        pred_exp_filtered_reward_rate = self.exp_filtered_reward_rate_prediction(actor_outputs)

        obs_pred_h = nn.relu(
            self.obs_pred_layer_1(actor_outputs)
        )
        obs_pred = self.obs_prediction(obs_pred_h)

        return logits, value, new_actor_hidden, new_critic_hidden, pred_env_quality, obs_pred, pred_exp_filtered_reward_rate
    
    
def init_network_and_params(
    hidden_size: int,
    action_size: int,
    obs_size: int,
    rnn_type: str,
    unit_noise_std: float,
    rng_key: jnp.ndarray,
):
    # Network input size: obs + prev_obs + prev_action + prev_reward
    input_size = obs_size + action_size + 1

    # Initialize network
    network = A2CRNNFlax(
        action_size=action_size,
        obs_size=obs_size,
        hidden_size=hidden_size, 
        rnn_type=rnn_type,
        unit_noise_std=unit_noise_std,
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