import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self, obs_size, n_actions):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128,128)
        self.actor_head = nn.Linear(128, n_actions) # Ham eylem değerlerini (logits) üretir.

        self.critic_head = nn.Linear(128, 1) # V(s)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor_head(x)

        action_probs = F.softmax(logits, dim=-1)

        state_value = self.critic_head(x) # Negatif değerler de gelebilir ortama göre
        
        return action_probs, state_value