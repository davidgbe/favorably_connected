import torch
import torch.nn as nn
import numpy as np


class GRU_RNN(nn.Module):

    def __init__(self, input_size, hidden_size, var_noise=0, device='cpu'):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.rnn = self.initialize_internal_structure(input_size, hidden_size).to(device)
        self.output_arm = nn.Linear(hidden_size, 1).to(device)
        # hidden states of [n_layers, n_envs, hidden_size]
        self.hidden_states = None
        self.var_noise = var_noise


    def initialize_internal_structure(self, input_size, hidden_size):
        rnn = nn.GRUCell(
            input_size=input_size, 
            hidden_size=hidden_size,
        )
        
        return rnn

    
    def reset_state(self):
        hidden_states = self.hidden_states.detach().cpu()
        self.hidden_states = None
        return hidden_states

    
    def set_state(self, hidden_states):
        self.hidden_states = hidden_states
        self.hidden_states = self.hidden_states.to(self.device)


    def forward(self, inputs):
        all_hidden = torch.zeros([inputs.shape[0], self.hidden_size, inputs.shape[2]])
        for k in np.arange(inputs.shape[2]):
            # if self.hidden_states is None, it just defaults to zeros
            if self.hidden_states is None: 
                new_hidden_states = self.rnn(inputs[..., k])
            else: 
                # need to add a dimension for num_layers
                new_hidden_states = self.rnn(inputs[..., k], self.hidden_states)

            # print('Hidden states mean', new_hidden_states.norm(1).mean().cpu())
            new_hidden_states += (self.var_noise**0.5) * torch.randn(new_hidden_states.shape).detach().to(self.device) # add gaussian noise to activity
            self.hidden_states = new_hidden_states
            all_hidden[..., k] = new_hidden_states

        output = self.output_arm(new_hidden_states)
        return (output.squeeze(1), all_hidden)