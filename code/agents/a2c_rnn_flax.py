from flax import linen as nn
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
            self.rnn_actor = nn.GRUCell(self.hidden_size)
            self.rnn_critic = nn.GRUCell(self.hidden_size)
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")
            
        # Actor and critic heads
        self.actor = nn.Dense(self.action_size)
        self.critic = nn.Dense(1)
        
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
        
        return logits, value, new_actor_hidden, new_critic_hidden