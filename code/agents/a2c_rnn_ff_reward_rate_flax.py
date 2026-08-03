"""A2C RNN variant where the actor is a GRU core + a feedforward readout head that
additionally receives the exponentially-filtered reward rate.

Architecture (vs. agents/a2c_rnn_flax.py):
  * The full input `x` carries the exponentially-filtered reward rate as its LAST column.
  * `x_main = x[..., :-1]` (everything except the reward rate) drives BOTH the actor GRU
    and the critic GRU.
  * Actor readout is a feedforward head: it takes [actor GRU output, reward_rate] and
    outputs the action logits.  (The reward rate is injected only here, not into the GRU.)
  * Critic outputs the value from its GRU output only (never sees the reward rate).

Drop-in: same class name (`A2CRNNFlax`), same `__call__` return tuple and same
`init_network_and_params` signature as agents/a2c_rnn_flax.py, EXCEPT the expected input
width is one larger (it now includes the trailing reward-rate column). Whatever builds the
network input must therefore append the filtered reward rate as the last feature.
"""

from flax import linen as nn
import jax.numpy as jnp
from jax import random
from typing import Optional
from agents.networks.vanilla_rnn_cell_flax import VanillaRNNCell


class A2CRNNFlax(nn.Module):
    """A2C network: actor = GRU core + feedforward head (with reward-rate input); critic = GRU."""
    action_size: int
    obs_size: int
    hidden_size: int
    rnn_type: str = 'GRU'  # 'VANILLA' or 'GRU'
    unit_noise_std: float = 1e-2
    init_scale: float = 1.0
    actor_ff_hidden_size: Optional[int] = None  # feedforward-head width (defaults to hidden_size)

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

        # Actor feedforward head: [actor GRU output, reward_rate] -> hidden -> action logits.
        ff_hidden = self.actor_ff_hidden_size if self.actor_ff_hidden_size is not None else self.hidden_size
        self.actor_ff_hidden = nn.Dense(
            ff_hidden,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )
        self.actor = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )

        # Critic head (from critic GRU output only).
        self.critic = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )

        # Auxiliary prediction heads (read from the actor GRU output), unchanged.
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
        # Scalar readout used only for supervised integration pretraining.
        self.integration_prediction = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(scale=self.init_scale),
        )

    def __call__(self, x, actor_hidden, critic_hidden):
        """
        Args:
            x: input (batch_size, input_dim); the LAST column is the exp-filtered reward rate.
            actor_hidden, critic_hidden: (batch_size, hidden_size)
        Returns:
            logits, value, new_actor_hidden, new_critic_hidden,
            pred_env_quality, obs_pred, pred_exp_filtered_reward_rate
        """
        # Everything except the trailing reward-rate column drives the recurrent cores.
        x_main = x[..., :-1]
        reward_rate = x[..., -1:]                      # (batch, 1)

        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x_main)
        new_critic_hidden, critic_outputs = self.rnn_critic(critic_hidden, x_main)

        # Apply noise to hidden states
        noise_actor = random.normal(self.make_rng('noise'), new_actor_hidden.shape) * self.unit_noise_std
        new_actor_hidden = new_actor_hidden + noise_actor
        noise_critic = random.normal(self.make_rng('noise'), new_critic_hidden.shape) * self.unit_noise_std
        new_critic_hidden = new_critic_hidden + noise_critic

        # Actor feedforward head: GRU output concatenated with the filtered reward rate -> logits.
        ff_in = jnp.concatenate([actor_outputs, reward_rate], axis=-1)
        ff_h = nn.relu(self.actor_ff_hidden(ff_in))
        logits = self.actor(ff_h)

        # Critic value from its GRU output only.
        value = self.critic(critic_outputs).squeeze(-1)

        # Auxiliary predictions from the actor GRU output (unchanged).
        pred_env_quality = self.env_quality_prediction(actor_outputs)
        pred_exp_filtered_reward_rate = self.exp_filtered_reward_rate_prediction(actor_outputs)
        obs_pred_h = nn.relu(self.obs_pred_layer_1(actor_outputs))
        obs_pred = self.obs_prediction(obs_pred_h)

        return (logits, value, new_actor_hidden, new_critic_hidden,
                pred_env_quality, obs_pred, pred_exp_filtered_reward_rate)

    def integrate(self, x, actor_hidden):
        """Actor-GRU forward with the scalar integration readout (supervised pretraining only).

        Splits off the trailing reward-rate column so the GRU sees the same x_main as __call__.
        """
        x_main = x[..., :-1]
        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x_main)
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
):
    # Input width: the caller supplies x whose LAST column is the exp-filtered reward rate; the
    # module routes x[..., :-1] to the GRUs and x[..., -1:] to the actor feedforward head. Here
    # x = [obs, prev_action_onehot, exp_filtered_reward_rate] -> obs_size + action_size + 1.
    # (If you also feed prev_reward into the GRU, add another +1 and widen the caller's input.)
    input_size = obs_size + action_size + 1

    network = A2CRNNFlax(
        action_size=action_size,
        obs_size=obs_size,
        hidden_size=hidden_size,
        rnn_type=rnn_type,
        unit_noise_std=unit_noise_std,
        init_scale=init_scale,
    )

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
