import torch
import torch.nn as nn
import torch.nn.functional as F


    
class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()


        # Boyut kontrolü (Tuple ise içindeki sayıyı al)
        obs_size = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
            
        # DDPG Makalesindeki Orijinal Katman Boyutları (400, 300)
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh() # Eylemleri kesin olarak -1 ile 1 arasına sıkıştırır
        )

    def forward(self, state):
        # Sigma (kararsızlık) yok. Doğrudan kesin eylemi (mu) döner.
        return self.net(state)
    


# D4PG için Critic Network  
class DistributionalCriticNetwork(nn.Module):
    # Yeni Parametreler: 
    # v_min ve v_max: Beklediğimiz minimum ve maksimum ödül sınırları
    # n_atoms: Histogramımızdaki çubuk sayısı (C51 makalesine göre 51 alıyoruz)
    def __init__(self, obs_shape, n_actions, v_min=-1000.0, v_max=2000.0, n_atoms=51):
        super().__init__()

        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms

        obs_size = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape

        self.net = nn.Sequential(
            nn.Linear(obs_size + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            #  DEĞİŞİM: Artık 1 sayı değil, 51 tane ihtimal üretiyoruz
            nn.Linear(300, n_atoms) 
        )
    
    def forward(self, state, action):
        # Girdileri birleştir
        x = torch.cat([state, action], dim=1)
        
        # Ağın ham çıktıları (Logits)
        logits = self.net(x)
        
        # DEĞİŞİM 2: Çıktıların bir İhtimal olabilmesi için hepsinin toplamının 1.0. Bu yüzden softmax kullanıyoruz
        probs = F.softmax(logits, dim=-1)
        
        return probs


""" --- DDPG için Critic Network ---
class CriticNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        
        obs_size = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
            
        # KRİTİK: Critic artık sadece State'i değil, Action'ı da girdi olarak alıyor
        self.net = nn.Sequential(
            nn.Linear(obs_size + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1) # Tek bir Q-Değeri (Kalite Puanı) döner
        )

    def forward(self, state, action):
        # State ve Action tensörlerini yan yana birleştir (Concatenate)
        # Örn: 17 boyutlu state + 6 boyutlu action = 23 boyutlu tek bir girdi vektörü
        x = torch.cat([state, action], dim=1)
        return self.net(x)"""
    


"""class ContinuousNetwork(nn.Module):

    def __init__(self, obs_shape, n_actions):
        super().__init__()

        # Minitaur'dan gelen gözlemler (eklem açıları, ivmeölçer vb.) 1 boyutludur
        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(256, n_actions)
        
        self.sigma = nn.Linear(256, n_actions)
        
        self.value1 = nn.Linear(256, 128)
        self.value2 = nn.Linear(128, 1)

    
    def forward(self, x):
        features = self.fc(x)
        
        mu = torch.tanh(self.mu(features))
    
        sigma = F.softplus(self.sigma(features)) + 1e-3

        values = self.value1(features)
        state_value = self.value2(values)
        
        return mu, sigma, state_value"""