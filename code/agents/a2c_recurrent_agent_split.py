import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class A2CRecurrentAgent:
    """
    Implements a recurrent A2C Agent
    Supports multple environments
    Implementation adapted from: https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/
    """
    def __init__(self, network, action_space_dims, n_envs, device="cpu", activity_weight=0,
                 critic_weight=0.05, entropy_weight=0.05, learning_rate=7e-4, gamma=0.9, optimizer=None,
                 input_noise_std=0):
        """
        Initializes recurrent agent: 
        Default values from Wang et al 2018
        Args: 
            network: a torch network to be used/trained
            action_space_dims: number of dims of action space,
            n_envs: number of environments
            device: str of cpu or cuda
            critic_weight: weight to assign the critic loss during training
            entropy_weight: weight to assign the entropy loss during training
            learning_rate: learning rate for optimizer
            gamma: discount factor
        """
        self.net = network
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight
        self.learning_rate = learning_rate  # Learning rate for policy optimization
        self.activity_weight = activity_weight
        self.gamma = gamma  # Discount factor
        self.action_space_dims = action_space_dims
        self.n_envs = n_envs
        self.device = device
        self.input_noise_std = input_noise_std
        self.log_probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.values = [] 
        self.entropies = []
        self.actions = []
        self.activities = []

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

    
    def sample_action(self, observations):
        """
        Given an observation, returns an action.
        Passes the observation, past action, past reward, and time as inputs into the network
        Additionally, keeps track of value, log probability of action, and entropy for future updates

        Args:
            observations: observations for this time [n_envs x obs_size]
        Returns:
            actions: Action indexes, with shape [n_envs, ] 
        """
        # ensures observations is a tensor
        obs = torch.tensor(observations).to(self.device)
        # grabs past actions and rewards
        past_action_one_hot = self.get_last_action_one_hot()
        past_reward = self.get_last_reward()

        # input will be [n_envs, (obs_size + action_size + 2)]
        # the 2 is for reward and time
        inputs = torch.cat((
            obs, past_action_one_hot, past_reward.unsqueeze(dim=1)
        ), dim=1)
        inputs += torch.normal(mean=torch.zeros_like(inputs), std=self.input_noise_std * torch.ones_like(inputs))
        inputs = inputs.to(self.device)

        # run through the network
        action_logits, value, hidden_unit_activity = self.net(inputs)

        # sample from action probs
        distrib = Categorical(logits=action_logits)
        action = distrib.sample()
        log_prob = distrib.log_prob(action)
        entropy = distrib.entropy()

        # keep track of values for future update
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.actions.append(action)
        self.entropies.append(entropy)
        self.activities.append(hidden_unit_activity)

        return action
    

    def get_last_action_one_hot(self):
        """
        Grabs a one-hot encoding representation of the last action
        Returns zeros if no previous action
        Returns: 
            tensor of [n_envs, n_action_space_dims]
        """
        past_action_one_hot = torch.zeros(self.n_envs, self.action_space_dims)
        if len(self.actions) > 0: 
            # grab that last action list, 
            last_actions = self.actions[-1]
            past_action_one_hot[:, last_actions] = 1
        return past_action_one_hot.to(self.device)
    

    def get_last_reward(self):
        """
        Grabs last reward, returns zeros if no previous rewards
        Returns: 
            tensor of [n_envs, ]
        """
        past_rewards = self.rewards[-1] if len(self.rewards) > 0 else torch.zeros(self.n_envs)
        return past_rewards.to(self.device)
    

    def append_reward(self, reward):
        """
        Adds a reward
        Args:
            reward: np array of [n_envs, ]
        """
        self.rewards.append(torch.tensor(reward))


    def get_losses(self):
        """
        Calculates the total loss, actor loss, critic loss, and entropy loss of the agent. 
        Total loss = actor_loss + critic_weight * critic loss + entropy_weight * entropy_loss
        See: Mnih et al., 2015: https://arxiv.org/abs/1602.01783
        """
        T = len(self.rewards)

        # need at least 2 trials before update makes sense
        if T < 2: 
            raise ValueError("Need at least 2 timesteps before computing losses")

        log_probs = torch.stack(self.log_probs).to(self.device)
        rewards = torch.stack(self.rewards).to(self.device)
        values = torch.stack(self.values).to(self.device)
        entropies = torch.stack(self.entropies).to(self.device)
        activities = torch.stack(self.activities)
        # make this an object attribute since it's useful for debugging later
        td_errs = torch.zeros((T, self.n_envs)).to(self.device)

        # step through time, calculate td errors (generally advantages)
        # NOTE: This is implementing vanilla actor critic,
        # TODO: For generalized impl that supports Monte Carlo methods as well, implement GAE
        # Schulman et al., 2015: https://arxiv.org/abs/1506.02438

        rolling_rewards = values[T-1, :].clone()

        for t in reversed(range(T-1)):
            rolling_rewards = self.gamma * rolling_rewards + rewards[t, :]
            td_errs[t, :] = rolling_rewards - values[t, :]

        # want to minimize TD error
        critic_loss = td_errs.pow(2).mean()   

        # A2C actor loss, derived from policy gradient theorem
        actor_loss = -(td_errs.detach() * log_probs).mean()

        # want to ENCOURAGE high entropy, so define negative loss
        entropy_loss = -entropies.mean()

        activity_loss = activities.pow(2).mean()

        total_loss = actor_loss + self.critic_weight * critic_loss + self.entropy_weight * entropy_loss + self.activity_weight * activity_loss
        return (total_loss, actor_loss, critic_loss, entropy_loss)


    def update(self, total_loss=None, retain_graph=False):
        """
        Updates network parameters via calculated total loss, calculates total 
        loss if it's not provided. 
        NOTE: update also resets all states, so as it's currently 
        implemented, this should only be called at the end of an episode. 
        """
        if total_loss is None:
            total_loss, _, _, _ = self.get_losses()
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        
        
    def reset_state(self, reset_hidden=True):
        """
        Resets the network, as well as the agent's states. 
        """
        hidden_states = None
        if reset_hidden:
            states = self.net.reset_state()
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []
        self.activities = []
        return states


    def get_state(self):
        states = self.net.reset_state()
        return states


    def set_state(self, *states):
        self.net.set_state(*states)


    def get_hidden_state_activities(self):
        return torch.stack(self.activities)

