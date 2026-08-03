"""A2C RNN variant with a multi-horizon (multistep-forecast) critic.

Identical to agents/a2c_rnn_flax.py EXCEPT the critic head outputs `value_horizon` values
per step instead of 1:

    value[b, h] = v_hat_{t+h | t}

i.e. the network's forecast, made at time t, of the value h steps ahead. This lets a k-step
bootstrap read BOTH v_t (h=0) and v_{t+k} (h=k) off a single time-t forecast.

Drop-in: same class name (`A2CRNNFlax`) and same __call__ return tuple as a2c_rnn_flax,
except `value` is (batch, value_horizon) instead of (batch,). `init_network_and_params`
takes an extra `value_horizon` argument.
"""

from flax import linen as nn
import jax.numpy as jnp
from jax import random
from agents.networks.vanilla_rnn_cell_flax import VanillaRNNCell


class A2CRNNFlax(nn.Module):
    """A2C RNN network with a multi-horizon value-forecast critic."""
    action_size: int
    obs_size: int
    hidden_size: int
    rnn_type: str = 'GRU'  # 'VANILLA' or 'GRU'
    unit_noise_std: float = 1e-2
    init_scale: float = 1.0
    value_horizon: int = 1  # number of forecast steps the critic emits per timestep

    def setup(self):
        if self.rnn_type == 'VANILLA':
            self.rnn_actor = VanillaRNNCell(self.hidden_size, self.unit_noise_std, self.init_scale)
            self.rnn_critic = VanillaRNNCell(self.hidden_size, self.unit_noise_std, self.init_scale)
        elif self.rnn_type == 'GRU':
            self.rnn_actor = nn.GRUCell(
                features=self.hidden_size,
                kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
            )
            self.rnn_critic = nn.GRUCell(
                features=self.hidden_size,
                kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
            )
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        self.actor = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        # Multi-horizon value head: value_horizon forecasts per timestep.
        self.critic = nn.Dense(
            self.value_horizon,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        self.env_quality_prediction = nn.Dense(
            3,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        self.exp_filtered_reward_rate_prediction = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        self.obs_pred_hidden_size = 16
        self.obs_pred_layer_1 = nn.Dense(
            self.obs_pred_hidden_size,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        self.obs_prediction = nn.Dense(
            self.obs_size + 1,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        self.integration_prediction = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )

    def __call__(self, x, actor_hidden, critic_hidden):
        """
        Returns:
            logits, value, new_actor_hidden, new_critic_hidden,
            pred_env_quality, obs_pred, pred_exp_filtered_reward_rate
        where value is (batch_size, value_horizon).
        """
        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x)
        new_critic_hidden, critic_outputs = self.rnn_critic(critic_hidden, x)

        noise_actor = random.normal(self.make_rng('noise'), new_actor_hidden.shape) * self.unit_noise_std
        new_actor_hidden = new_actor_hidden + noise_actor
        noise_critic = random.normal(self.make_rng('noise'), new_critic_hidden.shape) * self.unit_noise_std
        new_critic_hidden = new_critic_hidden + noise_critic

        logits = self.actor(actor_outputs)
        value = self.critic(critic_outputs)          # (batch, value_horizon) -- NOT squeezed

        pred_env_quality = self.env_quality_prediction(actor_outputs)
        pred_exp_filtered_reward_rate = self.exp_filtered_reward_rate_prediction(actor_outputs)
        obs_pred_h = nn.relu(self.obs_pred_layer_1(actor_outputs))
        obs_pred = self.obs_prediction(obs_pred_h)

        return (logits, value, new_actor_hidden, new_critic_hidden,
                pred_env_quality, obs_pred, pred_exp_filtered_reward_rate)

    def integrate(self, x, actor_hidden):
        """Actor-GRU forward with the scalar integration readout (supervised pretraining only)."""
        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x)
        noise_actor = random.normal(self.make_rng('noise'), new_actor_hidden.shape) * self.unit_noise_std
        new_actor_hidden = new_actor_hidden + noise_actor
        pred = self.integration_prediction(actor_outputs)
        return pred, new_actor_hidden


def init_network_and_params(
    hidden_size: int,
    action_size: int,
    obs_size: int,
    rnn_type: str,
    unit_noise_std: float,
    rng_key: jnp.ndarray,
    init_scale: float = 1.0,
    value_horizon: int = 1,
):
    input_size = obs_size + action_size + 1  # obs + prev_action_onehot + prev_reward

    network = A2CRNNFlax(
        action_size=action_size,
        obs_size=obs_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        unit_noise_std=unit_noise_std,
        init_scale=init_scale,
        value_horizon=value_horizon,
    )

    param_key, hidden_key = random.split(rng_key, 2)
    dummy_input = jnp.zeros((1, input_size))
    dummy_hidden = jnp.zeros((1, hidden_size))

    params = network.init(param_key, dummy_input, dummy_hidden, dummy_hidden)

    return network, params
