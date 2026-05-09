import torch
import torch.nn as nn
import numpy as np

class A2C_Critic(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        # Critic sadece State'i alır ve tek bir sayı V(s) üretir
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Çıktı: Durumun Değeri V(s)
        )

    def forward(self, x):
        return self.net(x)


class A2C_Actor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        
        # Ağ sadece merkez noktasını (mu) tahmin eder
        self.mu_net = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Tanh() # Motorlara -1 ile 1 arası güç gideceği için Tanh kullanıyoruz
        )
        
        # Log-Standart Sapma: Bağımsız, ağdan ayrı ama Optimizer tarafından eğitilebilir (requires_grad=True)
        # Başlangıçta 0 yapıyoruz. exp(0) = 1 olduğu için başlangıç standart sapmamız 1'dir.
        self.logstd = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        # Duruma göre merkezi (mu) bul
        mu = self.mu_net(x)
        
        # Eğitimde olan logstd'yi dışarı aktar (Batches için boyutunu eşitleyerek)
        # expand_as, logstd'nin boyutunu batch boyutuna (mu'nunkine) genişletir
        logstd = self.logstd.expand_as(mu)
        
        # Gerçek standart sapmayı (sigma) logaritmadan kurtararak (exp) hesapla
        std = torch.exp(logstd)
        
        return mu, std