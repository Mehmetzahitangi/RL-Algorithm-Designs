import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(PPONetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
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
        """Simülasyonda gezerken kullanılır (Rollout toplamak için)"""
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), self.critic(state)
    
    def evaluate(self, state, action):
        """Eğitim (Update) sırasında eski hareketleri değerlendirmek için kullanılır"""
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        # Log ihtimali, entropi (rastgelelik) ve Q-değeri
        return dist.log_prob(action), dist.entropy(), self.critic(state)