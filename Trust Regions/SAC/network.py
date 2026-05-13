import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Matematiksel güvenlik sınırları (Ağın patlamaması için)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# 1. İKİZ ELEŞTİRMEN (TWIN CRITIC)
class SAC_Critic(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(SAC_Critic, self).__init__()
        
        # SAC, A2C/PPO gibi "Durum Değeri V(s)" değil, DDPG gibi "Hareket Değeri Q(s,a)" üretir.
        # Bu yüzden giriş boyutu (Durum + Aksiyon) kadardır.
        
        # Q1 Ağı
        self.q1_net = nn.Sequential(
            nn.Linear(obs_shape[0] + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 Ağı (İkizi)
        self.q2_net = nn.Sequential(
            nn.Linear(obs_shape[0] + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Durum ve Aksiyon vektörlerini yan yana yapıştır
        sa = torch.cat([state, action], 1)
        
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2


# 2. KAOTİK AKTÖR (SOFT ACTOR)
class SAC_Actor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(SAC_Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(256, n_actions)
        self.log_std_layer = nn.Linear(256, n_actions)

    def forward(self, state):
        features = self.net(state)
        mu = self.mu_layer(features)
        log_std = self.log_std_layer(features)
        
        # Ajan bazen çok emin (std=0) veya çok kaotik (std=sonsuz) olmak ister.
        # Motorun çökmemesi için bu kaosu fiziksel sınırlara kelepçeliyoruz.
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # 1. SİHİR: REPARAMETERIZATION TRICK (Türev alınabilen rastgelelik)
        # sample() yerine rsample() kullanıyoruz. 
        # Matematiksel karşılığı: x_t = mu + std * N(0, 1)
        x_t = dist.rsample() 
        
        # 2. Fiziksel Motor Sınırları (-1 ile 1 arasına basmak için Tanh)
        y_t = torch.tanh(x_t)
        action = y_t
        
        # 3. MATEMATİKSEL BEDEL (Tanh Düzeltmesi)
        # Tanh fonksiyonu olasılık uzayını büker. Köşelere gidildikçe yoğunluk artar.
        # Doğru Entropiyi bulabilmek için bu bükülmenin türevini olasılıktan çıkarmalıyız.
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Eğitim için (action ve log_prob), Test için (saf tanh(mu)) döndür
        return action, log_prob, torch.tanh(mu)