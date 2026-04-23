import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        """
        Modelin iskeletini (katmanlarını) tanımladığımız yer.
        obs_size: Gözlem uzayının boyutu (Örn: CartPole için 4)
        n_actions: Seçilebilecek olası eylem sayısı (Örn: CartPole için 2)
        """
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256,256)
        self.action_head = nn.Linear(256, n_actions) # Ham eylem değerlerini (logits) üretir.

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)

        # dim=-1 işlemin son boyutta (eylemler üzerinde) yapılmasını sağlar.
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs