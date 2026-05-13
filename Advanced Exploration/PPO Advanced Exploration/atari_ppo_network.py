import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from noisy_layers import NoisyLinear # Kendi motorumuz!

class AtariNoisyPPONetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariNoisyPPONetwork, self).__init__()
        
        # ORTAK GÖZ (CNN Katmanları): Piksellerden anlam çıkartır
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # CNN'in çıkış boyutunu hesapla (84x84 için genellikle 3136 çıkar)
        conv_out_size = self._get_conv_out(input_shape)
        
        # AKTÖR (Pilot): Hangi tuşa basmalıyım? (GÜRÜLTÜLÜ/MERAKLI)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions) # Zarlar burada atılıyor!
        )
        
        # ELEŞTİRMEN (Hakem): Bu ekran kaç puan eder? (GÜRÜLTÜSÜZ)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1) # Flatten (Düzleştir)
        return self.actor(conv_out), self.critic(conv_out)

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
    
    def evaluate(self, state, action):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value

    def sample_noise(self):
        """PPO yığını (batch) başlamadan önce ajana yeni bir merak profili yükler."""
        for m in self.actor.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()