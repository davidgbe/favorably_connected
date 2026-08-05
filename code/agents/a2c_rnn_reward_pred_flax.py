"""A2C RNN with a k-window reward-predictor critic (for RPE-based credit assignment).

Actor: GRU policy -> logits (unchanged from a2c_rnn_flax). The "critic" here is NOT a value
head but a small feedforward reward predictor: given a flattened k-step window of
(observation, self-action) pairs -- NO rewards -- it predicts the immediate reward r_t. The
reward-prediction error r_t - r_hat_t is the basis for credit (see compute_a2c_loss).

__call__ keeps the same 7-tuple return as a2c_rnn_flax (a dummy scalar value head is kept for
API compatibility with collect_trajectory but is unused). The reward predictor is applied
separately via the `predict_reward` method (called in the loss on windows built from the
stored trajectory). `init_all` exercises both so init creates every submodule's params.
"""

from flax import linen as nn
import jax.numpy as jnp
from jax import random
from agents.networks.vanilla_rnn_cell_flax import VanillaRNNCell


class A2CRNNFlax(nn.Module):
    action_size: int
    obs_size: int
    hidden_size: int
    rnn_type: str = 'GRU'
    unit_noise_std: float = 1e-2
    init_scale: float = 1.0
    reward_pred_hidden_size: int = 12
    reward_pred_init_scale: int = 0.001

    def setup(self):
        if self.rnn_type == 'VANILLA':
            self.rnn_actor = VanillaRNNCell(self.hidden_size, self.unit_noise_std, self.init_scale)
            self.rnn_critic = VanillaRNNCell(self.hidden_size, self.unit_noise_std, self.init_scale)
        elif self.rnn_type == 'GRU':
            self.rnn_actor = nn.GRUCell(features=self.hidden_size,
                                        kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
            self.rnn_critic = nn.GRUCell(features=self.hidden_size,
                                         kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        self.actor = nn.Dense(self.action_size, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.critic = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))  # unused
        self.env_quality_prediction = nn.Dense(3, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.exp_filtered_reward_rate_prediction = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.obs_pred_hidden_size = 16
        self.obs_pred_layer_1 = nn.Dense(self.obs_pred_hidden_size, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.obs_prediction = nn.Dense(self.obs_size + 1, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.integration_prediction = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))

        # k-window reward-predictor MLP (input = flattened [obs, self-action] window).
        self.reward_pred_l1 = nn.Dense(self.reward_pred_hidden_size, kernel_init=nn.initializers.orthogonal(scale=self.reward_pred_init_scale))
        self.reward_pred_l2 = nn.Dense(self.reward_pred_hidden_size, kernel_init=nn.initializers.orthogonal(scale=self.reward_pred_init_scale))
        self.reward_pred_out = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=self.reward_pred_init_scale))

    def __call__(self, x, actor_hidden, critic_hidden):
        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x)
        new_critic_hidden, critic_outputs = self.rnn_critic(critic_hidden, x)

        noise_actor = random.normal(self.make_rng('noise'), new_actor_hidden.shape) * self.unit_noise_std
        new_actor_hidden = new_actor_hidden + noise_actor
        noise_critic = random.normal(self.make_rng('noise'), new_critic_hidden.shape) * self.unit_noise_std
        new_critic_hidden = new_critic_hidden + noise_critic

        logits = self.actor(actor_outputs)
        value = self.critic(critic_outputs).squeeze(-1)               # unused by the RPE loss
        pred_env_quality = self.env_quality_prediction(actor_outputs)
        pred_exp_filtered_reward_rate = self.exp_filtered_reward_rate_prediction(actor_outputs)
        obs_pred = self.obs_prediction(nn.relu(self.obs_pred_layer_1(actor_outputs)))
        return (logits, value, new_actor_hidden, new_critic_hidden,
                pred_env_quality, obs_pred, pred_exp_filtered_reward_rate)

    def predict_reward(self, window):
        """window: (..., feat) reward-predictor input. As used by the conditional critic this is
        [flattened k-window of obs ending at s, anchor-action one-hot, normalized lag j], and the
        output is the reward predicted at s given that action was taken j steps earlier.
        Generic MLP over the last axis. Returns (...,) predicted reward."""
        h = nn.relu(self.reward_pred_l1(window))
        h = nn.relu(self.reward_pred_l2(h))
        return self.reward_pred_out(h).squeeze(-1)

    def init_all(self, x, actor_hidden, critic_hidden, window):
        """Exercise every submodule so init creates all params (actor + reward predictor)."""
        self.__call__(x, actor_hidden, critic_hidden)
        self.predict_reward(window)
        return 0


def init_network_and_params(hidden_size, action_size, obs_size, rnn_type, unit_noise_std,
                            rng_key, init_scale=1.0, reward_pred_hidden_size=64, window_dim=None):
    input_size = obs_size + action_size + 1
    if window_dim is None:
        window_dim = obs_size + action_size            # k=1 fallback
    network = A2CRNNFlax(action_size=action_size, obs_size=obs_size, hidden_size=hidden_size,
                         rnn_type=rnn_type, unit_noise_std=unit_noise_std, init_scale=init_scale,
                         reward_pred_hidden_size=reward_pred_hidden_size)
    param_key, noise_key = random.split(rng_key, 2)
    dummy_input = jnp.zeros((1, input_size))
    dummy_hidden = jnp.zeros((1, hidden_size))
    dummy_window = jnp.zeros((1, window_dim))
    params = network.init({'params': param_key, 'noise': noise_key},
                          dummy_input, dummy_hidden, dummy_hidden, dummy_window,
                          method=A2CRNNFlax.init_all)
    return network, params
