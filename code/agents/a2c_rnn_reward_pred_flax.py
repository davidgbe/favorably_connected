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
    belief_dim: int = 8             # dimensionality of the critic belief vector b_t (context for S_C)
    belief_pred_hidden: int = 64    # feedforward hidden size for the belief -> future [obs, reward] predictor
    belief_pred_horizon: int = 20   # N_BELIEF_PREDICT: number of future [obs, reward] tuples the belief predicts

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

        # belief head (on the CRITIC): a hidden_size "belief" vector read off the critic RNN output,
        # plus a feedforward net predicting the next belief_pred_horizon [OBSERVATION, REWARD] tuples
        # from it ((obs_size + 1) dims each, flattened). Belief similarity S_C(b_l, b_t) gates M by context.
        self.belief_readout = nn.Dense(self.belief_dim, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.belief_pred_l1 = nn.Dense(self.belief_pred_hidden, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))
        self.belief_pred_out = nn.Dense(self.belief_pred_horizon * (self.obs_size + 1),
                                        kernel_init=nn.initializers.orthogonal(scale=self.init_scale))

        # dedicated 1-step-ahead reward predictor E[r_t] on the critic output -- a strong, un-swamped
        # reward signal (its own MSE, not diluted by the belief's obs channels) that forces the critic
        # RNN to track within-patch depletion. reff_pg's M uses this as E[r].
        self.reward_pred_1step_head = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=self.init_scale))

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
        # belief vector read off the critic RNN output (its "context" representation for S_C), UNIT-
        # NORMALIZED so all information is in the angle: the belief predictor (predict_belief) then can't
        # cheat via magnitude and must encode context in the DIRECTION -- which is what S_C compares.
        belief = self.belief_readout(critic_outputs)
        belief = belief / (jnp.linalg.norm(belief, axis=-1, keepdims=True) + 1e-8)
        # belief_pred: the belief's forecast of the next belief_pred_horizon [obs, reward] tuples
        # (flattened). Its REWARD channels are the reward forecast E[r_{t+j}] used by the loss/M -- this
        # REPLACES the old actor reward_pred head, so the reward prediction now lives on the critic.
        belief_pred = self.predict_belief(belief)
        # dedicated 1-step reward forecast E[r_t] (critic-side, its own head/loss).
        reward_pred_1step = self.reward_pred_1step_head(critic_outputs).squeeze(-1)
        return (logits, value, new_actor_hidden, new_critic_hidden,
                pred_env_quality, obs_pred, pred_exp_filtered_reward_rate, belief_pred, belief, reward_pred_1step)

    def predict_belief(self, belief):
        """Feedforward predictor: belief (..., hidden_size) -> flattened next belief_pred_horizon
        [observation, reward] tuples ((obs_size + 1) dims each). Trained by MSE; its gradient shapes
        belief_readout and the critic's recurrent weights. Reshape to (..., horizon, obs_size + 1)."""
        return self.belief_pred_out(nn.relu(self.belief_pred_l1(belief)))

    def predict_next(self, hidden):
        """1-step forward prediction from a hidden state: next observation + reward.
        hidden: (..., hidden_size). Returns (..., obs_size + 1) = [pred_obs, pred_reward]
        (reuses the obs_prediction MLP head)."""
        return self.obs_prediction(nn.relu(self.obs_pred_layer_1(hidden)))

    def init_all(self, x, actor_hidden, critic_hidden):
        """Exercise every submodule so init creates all params (actor + reward readout).
        `window` is unused (kept for signature compatibility); the readout is on the hidden state."""
        self.__call__(x, actor_hidden, critic_hidden)   # exercises belief_readout + predict_belief
        self.predict_next(critic_hidden)
        return 0


def init_network_and_params(hidden_size, action_size, obs_size, rnn_type, unit_noise_std,
                            rng_key, init_scale=1.0, belief_dim=8, belief_pred_hidden=64,
                            belief_pred_horizon=20):
    input_size = obs_size + action_size + 1
    network = A2CRNNFlax(action_size=action_size, obs_size=obs_size, hidden_size=hidden_size,
                         rnn_type=rnn_type, unit_noise_std=unit_noise_std, init_scale=init_scale,
                         belief_dim=belief_dim, belief_pred_hidden=belief_pred_hidden,
                         belief_pred_horizon=belief_pred_horizon)
    param_key, noise_key = random.split(rng_key, 2)
    dummy_input = jnp.zeros((1, input_size))
    dummy_hidden = jnp.zeros((1, hidden_size))
    params = network.init({'params': param_key, 'noise': noise_key},
                          dummy_input, dummy_hidden, dummy_hidden,
                          method=A2CRNNFlax.init_all)
    return network, params
