import torch
import torch.nn as nn
import numpy as np


class GRU_RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, var_noise=0, device='cpu'):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.rnn = self.initialize_internal_structure(input_size, hidden_size).to(device)
        self.output_arm = nn.Linear(hidden_size, output_size).to(device)
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


    def forward(self, inputs, output_steps=None, stateful=True):
        if output_steps is None:
            output_steps = inputs.shape[2]

        output = torch.empty((inputs.shape[0], 1, output_steps)).to(self.device)
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
            if stateful:
                self.hidden_states = new_hidden_states
            all_hidden[..., k] = new_hidden_states

            if k >= inputs.shape[2] - output_steps:
                output[..., k - (inputs.shape[2] - output_steps)] = self.output_arm(new_hidden_states)
        return (output.squeeze(1), all_hidden)


    def turn_off_grad(self):
        for param in self.rnn.parameters():
            param.requires_grad = False
        for param in self.output_arm.parameters():
            param.requires_grad = False
        