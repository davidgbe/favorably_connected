import torch
import torch.nn as nn
import numpy as np


class A2CRNN(nn.Module):
    """
    A recurrent network with two linear readouts output arms for action and value
    Can be used for A2C but also for REINFORCE with Baseline
    Drawing mainly from 
    https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/ , 
    https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py , and
    Want et al., 2018
    Adjusted for the WCST
    """

    def __init__(self, input_size, action_size, hidden_size, var_noise=0, device='cpu'):
        """Initializes a neural network that estimates the logits to a 
        categorical action distrbution, as well as the value of that state


        Args:
            input_size: Dimensions of the inputs. 
            action_size: Dimensions of the action space
            hidden_size: Dimensions of any hidden layers
            device: device for network to live on. 
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.rnn = nn.RNNCell(
            input_size=self.input_size, 
            hidden_size=hidden_size,
            nonlinearity='relu',
        ).to(device)
        self.critic_rnn = nn.RNNCell(
            input_size=self.input_size, 
            hidden_size=hidden_size,
            nonlinearity='relu',
        ).to(device)
        self.action_arm = nn.Linear(hidden_size, action_size).to(device)
        self.value_arm = nn.Linear(hidden_size, 1).to(device)
        # hidden states of [n_layers, n_envs, hidden_size]
        self.hidden_states = None
        self.critic_hidden_states = None
        self.var_noise = var_noise
    
    def reset_state(self):
        hidden_states = self.hidden_states.detach().cpu()
        critic_hidden_states = self.critic_hidden_states.detach().cpu()
        self.hidden_states = None
        self.critic_hidden_states = None
        return hidden_states, critic_hidden_states

    
    def set_state(self, hidden_states, critic_hidden_states):
        self.hidden_states = hidden_states
        self.hidden_states = self.hidden_states.to(self.device)
        self.critic_hidden_states = critic_hidden_states
        self.critic_hidden_states = self.critic_hidden_states.to(self.device)


    def forward(self, inputs):
        """
        Returns actions logits for each env
        Args: 
            inputs: tensor of [n_envs x input_size]
        Returns:
            action_logits: A tensor with the action logits, with shape [n_envs, n_actions] 
            state_values: A tensor with the state values, with shape [n_envs,].
        """

        hidden_state_noise = (self.var_noise**0.5) * torch.randn((self.hidden_size)).detach().to(self.device) # add gaussian noise to activity
        critic_hidden_state_noise = (self.var_noise**0.5) * torch.randn((self.hidden_size)).detach().to(self.device) # add gaussian noise to activity

        # if self.hidden_states is None, it just defaults to zeros
        if self.hidden_states is None: 
            self.hidden_states = self.rnn(inputs) + hidden_state_noise
            self.critic_hidden_states = self.critic_rnn(inputs) + critic_hidden_state_noise
        else: 
            # need to add a dimension for num_layers
            self.hidden_states = self.rnn(inputs, self.hidden_states) + hidden_state_noise
            self.critic_hidden_states = self.critic_rnn(inputs, self.critic_hidden_states) + critic_hidden_state_noise

        # need to get rid of layers dimension, which is at dim 0. 
        action_logits = self.action_arm(self.hidden_states)
        value = self.value_arm(self.critic_hidden_states)
        # values have 2nd dimension of 1, get rid of it
        return (action_logits, value.squeeze(1), self.hidden_states.clone())