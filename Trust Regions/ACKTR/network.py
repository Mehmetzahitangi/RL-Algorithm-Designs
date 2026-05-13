import torch
import torch.nn as nn
import torch.nn.functional as F

class ACKTR_Actor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(ACKTR_Actor, self).__init__()
        # K-FAC bu Linear katmanlara otomatik olarak sızıp matrisleri oluşturacak
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Ortalama (mu) ve Standart Sapma (std) için çıkış katmanları
        self.mu_head = nn.Linear(64, n_actions)
        self.std_head = nn.Linear(64, n_actions)

    def forward(self, x):
        features = self.net(x)
        mu = self.mu_head(features)
        
        # Softplus: Ağın ürettiği değerleri her zaman 0'dan büyük (pozitif) tutar.
        # Standart sapma negatif olamayacağı için bu matematiksel bir zorunluluktur.
        # 1e-5 ekliyoruz ki tamamen 0 olup varyansı patlatmasın (Sıfıra bölünme hatası).
        std = F.softplus(self.std_head(features)) + 1e-5
        
        return mu, std

class ACKTR_Critic(nn.Module):
    def __init__(self, obs_shape):
        super(ACKTR_Critic, self).__init__()
        # Eleştirmen sadece gidişata bakıp TEK BİR puan (Value) üretir
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)