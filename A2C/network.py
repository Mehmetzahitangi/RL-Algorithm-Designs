import torch
import torch.nn as nn
import torch.nn.functional as F



class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_shape, n_actions):
        super(ActorCriticNetwork, self).__init__()

        # Modüler Kontrol: Girdi 1D (sayı/vektör) mu yoksa 3D (resim) mi?
        # Kullanıcı yanlışlıkla tek sayı girerse onu tuple yap (örn: 8 -> (8,))
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
            
        self.is_image = len(obs_shape) == 3

        if self.is_image:
            # --- CNN GÖVDESİ (Pong vb. için) ---
            # Beklenen girdi: (Batch_Size, 4, 84, 84)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Dinamik boyut hesaplama
            # 84x84 resim bu katmanlardan geçince genellikle 3136 (64*7*7) boyutuna iner.
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                conv_out_size = self.feature_extractor(dummy).shape[1]
            
            self.fc = nn.Linear(conv_out_size, 512)
            feature_size = 512

        else:
            # --- MLP GÖVDESİ (CartPole, LunarLander için) ---
            self.feature_extractor = nn.Sequential(
                nn.Linear(obs_shape[0], 128),
                nn.ReLU()
            )
            self.fc = nn.Identity() # Ekstra katmana gerek yok, içinden direkt geçer
            feature_size = 128

        # --- Shared Heads ---
        # Hem CNN'den hem MLP'den gelen veriler en son buradan çıkar
        self.actor = nn.Linear(feature_size, n_actions)
        self.critic = nn.Linear(feature_size, 1)

        #self.fc1 = nn.Linear(obs_size, 128)
        #self.fc2 = nn.Linear(128,128)
        #self.actor_head = nn.Linear(128, n_actions) # Ham eylem değerlerini (logits) üretir.

        #self.critic_head = nn.Linear(128, 1) # V(s)


    def forward(self, x):
        # Ortak veri akışı
        features = self.feature_extractor(x)
        
        if self.is_image:
            features = F.relu(self.fc(features))
            
        # Olasılıklar ve Durum Değeri
        action_probs = F.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features) # Negatif değerler de gelebilir ortama göre
        
        return action_probs, state_value
    
    #def forward(self, x):
    #
    #    x = F.relu(self.fc1(x))
    #    x = F.relu(self.fc2(x))
    #    logits = self.actor_head(x)
    #
    #    action_probs = F.softmax(logits, dim=-1)
    #
    #    state_value = self.critic_head(x) # Negatif değerler de gelebilir ortama göre
    #    
    #    return action_probs, state_value