import torch
import torch.nn as nn
from noisy_layers import NoisyLinear

class NoisyDQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(NoisyDQN, self).__init__()
        
        # Lapan'ın MountainCar Mimarisi: Normal Linear -> ReLU -> NoisyLinear
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            NoisyLinear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def sample_noise(self):
        """
        Ağın içindeki tüm NoisyLinear katmanlarını bulup
        içlerindeki gürültüyü yeniden örnekler (Matematiksel Zarları yeniden atar).
        """
        for module in self.net.modules():
            if isinstance(module, NoisyLinear):
                module.sample_noise()