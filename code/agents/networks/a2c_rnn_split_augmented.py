import torch
import torch.nn as nn
import numpy as np


class A2CRNNAugmented(nn.Module):
    """
    A recurrent network with two linear readouts output arms for action and value
    Can be used for A2C but also for REINFORCE with Baseline
    Drawing mainly from 
    https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/ , 
    https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py , and
    Want et al., 2018
    Adjusted for the WCST
    """

    def __init__(self, subnetwork, input_size, action_size, hidden_size, var_noise=0, device='cpu'):
        """Initializes a neural network that estimates the logits to a 
        categorical action distrbution, as well as the value of that state


        Args:
            input_size: Dimensions of the inputs. 
            action_size: Dimensions of the action space
            hidden_size: Dimensions of any hidden layers
            device: device for network to live on. 
        """
        super().__init__()

        self.subnetwork = subnetwork
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.rnn = nn.GRUCell(
            input_size=self.input_size + subnetwork.output_size, 
            hidden_size=hidden_size,
        ).to(device)
        self.critic_rnn = nn.GRUCell(
            input_size=self.input_size, 
            hidden_size=hidden_size,
        ).to(device)
        self.proj_to_subnetwork = nn.Linear(hidden_size, subnetwork.input_size).to(device)
        self.action_arm = nn.Linear(hidden_size, action_size).to(device)
        self.value_arm = nn.Linear(hidden_size, 1).to(device)
        # hidden states of [n_layers, n_envs, hidden_size]
        self.hidden_states = None
        self.critic_hidden_states = None
        self.var_noise = var_noise
    
    def reset_state(self):
        subnetwork_hidden_states = self.subnetwork.reset_state()
        hidden_states = self.hidden_states.detach().cpu()
        critic_hidden_states = self.critic_hidden_states.detach().cpu()
        self.hidden_states = None
        self.critic_hidden_states = None
        return hidden_states, critic_hidden_states, subnetwork_hidden_states

    
    def set_state(self, hidden_states, critic_hidden_states, subnetwork_hidden_states):
        self.hidden_states = hidden_states
        self.hidden_states = self.hidden_states.to(self.device)
        self.critic_hidden_states = critic_hidden_states
        self.critic_hidden_states = self.critic_hidden_states.to(self.device)
        self.subnetwork.set_state(subnetwork_hidden_states)


    def forward(self, inputs):
        """
        Returns actions logits for each env
        Args: 
            inputs: tensor of [n_envs x input_size]
        Returns:
            action_logits: A tensor with the action logits, with shape [n_envs, n_actions] 
            state_values: A tensor with the state values, with shape [n_envs,].
        """
        # if self.hidden_states is None, it just defaults to zeros
        if self.hidden_states is None:
            new_inputs_to_subnetwork = torch.zeros(inputs.shape[0], self.subnetwork.input_size).to(self.device)
            new_inputs_from_subnetwork, _ = self.subnetwork(new_inputs_to_subnetwork.unsqueeze(2))

            augmented_inputs = torch.cat((inputs, new_inputs_from_subnetwork), dim=1)
            new_hidden_states = self.rnn(augmented_inputs)
            new_critic_hidden_states = self.critic_rnn(inputs)
        else:
            new_inputs_to_subnetwork = self.proj_to_subnetwork(self.hidden_states)
            new_inputs_from_subnetwork, _ = self.subnetwork(new_inputs_to_subnetwork.unsqueeze(2))
            # need to add a dimension for num_layers
            augmented_inputs = torch.cat((inputs, new_inputs_from_subnetwork), dim=1)
            new_hidden_states = self.rnn(augmented_inputs, self.hidden_states)
            new_critic_hidden_states = self.critic_rnn(inputs, self.critic_hidden_states)

        # print('Hidden states mean', new_hidden_states.norm(1).mean().cpu())
        new_hidden_states += (self.var_noise**0.5) * torch.randn(new_hidden_states.shape).detach().to(self.device) # add gaussian noise to activity
        new_critic_hidden_states += (self.var_noise**0.5) * torch.randn(new_critic_hidden_states.shape).detach().to(self.device) # add gaussian noise to activity
        self.hidden_states = new_hidden_states
        self.critic_hidden_states = new_critic_hidden_states
        # need to get rid of layers dimension, which is at dim 0. 
        action_logits = self.action_arm(new_hidden_states)
        value = self.value_arm(new_critic_hidden_states)

        extended_new_hidden_states = torch.cat((
            new_hidden_states.clone(),
            self.subnetwork.hidden_states.clone(),
        ), dim=1)
        # values have 2nd dimension of 1, get rid of it
        return (action_logits, value.squeeze(1), extended_new_hidden_states)