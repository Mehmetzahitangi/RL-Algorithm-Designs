import torch
import torch.nn as nn
from torch.distributions import Categorical
from noisy_layers import NoisyLinear

class NoisyPPONetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(NoisyPPONetwork, self).__init__()
        
        #  (Sadece son katmanı Noisy yapıyoruz)
        self.actor = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            NoisyLinear(64, n_actions) 
        )
        
        # (Temiz ve gürültüsüz kalmalı )
        self.critic = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, state):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(state)
    
    def evaluate(self, state, action):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), self.critic(state)

    def sample_noise(self):
        """Aktörün içindeki NoisyLinear katmanlarının zarlarını yeniden atar"""
        for m in self.actor.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()