import torch
import torch.nn as nn
import torch.nn.functional as F

class I2A_Agent(nn.Module):
    def __init__(self, obs_shape, n_actions, env_model, rollout_policy, rollout_encoder, rollout_steps=5):
        super(I2A_Agent, self).__init__()
        self.n_actions = n_actions
        self.rollout_steps = rollout_steps
        
        # İçine yerleştirilen alt beyinler (Alt modeller)
        self.env_model = env_model
        self.rollout_policy = rollout_policy
        self.rollout_encoder = rollout_encoder
        
        # MODEL-FREE (İçgüdüsel Beyin)
        # O anki gerçek ekranı işleyen standart CNN
        self.mf_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        mf_conv_out_size = 3136 # Atari (84x84) için özellik boyutu
        
        # BÜYÜK BİRLEŞME KATMANI (Model-Free + Hayal Gücü)
        # Her bir olası aksiyon için encoder'dan 256 uzunluğunda bir özet gelecek.
        # Toplam Hayal Gücü Boyutu = n_actions * 256
        # Birleşmiş Boyut = İçgüdüsel Boyut + Hayal Gücü Boyutu
        fc_input_size = mf_conv_out_size + (n_actions * 256)
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU()
        )
        
        # AKTÖR VE ELEŞTİRMEN (Son Karar Headerları)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        
        #  ADIM 1: İçgüdüsel (Model-Free) Özellikleri Çıkar 
        mf_features = self.mf_conv(x).view(batch_size, -1)
        
        #  ADIM 2: Hayal Kurma (Imagination) Süreci 
        imagination_features = []
        
        # Ajan, yapabileceği HER BİR aksiyon için ayrı ayrı hayal kurar
        for action_idx in range(self.n_actions):
            # Olası aksiyonu One-Hot vektöre çevir (Örn: [0, 1, 0, 0])
            action_one_hot = torch.zeros(batch_size, self.n_actions).to(x.device)
            action_one_hot[:, action_idx] = 1.0
            
            obs_seq = []
            reward_seq = []
            
            curr_obs = x
            curr_action = action_one_hot
            
            # Belirlenen adım sayısı kadar (Örn: 5) geleceğe git
            for step in range(self.rollout_steps):
                # EM'ye sor: "Eğer bu tuşa basarsam ne olacak?"
                next_obs, reward = self.env_model(curr_obs, curr_action)
                
                obs_seq.append(next_obs.unsqueeze(1)) # [Batch, 1, Kanal, H, W]
                reward_seq.append(reward.unsqueeze(1)) # [Batch, 1, 1]
                
                curr_obs = next_obs
                
                # Bir sonraki hamleyi ana beyin değil, küçük Taklitçi Pilot seçsin
                if step < self.rollout_steps - 1:
                    policy_logits = self.rollout_policy(curr_obs)
                    next_action_idx = policy_logits.argmax(dim=1) # En mantıklı hamleyi al
                    
                    curr_action = torch.zeros(batch_size, self.n_actions).to(x.device)
                    curr_action.scatter_(1, next_action_idx.unsqueeze(1), 1.0)
                    
            # Adımları zaman ekseninde birleştir
            obs_seq = torch.cat(obs_seq, dim=1)
            reward_seq = torch.cat(reward_seq, dim=1)
            
            # Hayal Kodlayıcıya (LSTM/GRU) ver ve bu uzun rüyanın "özetini" al!
            action_summary = self.rollout_encoder(obs_seq, reward_seq) # [Batch, 256]
            imagination_features.append(action_summary)
            
        # Bütün aksiyonların hayal özetlerini yan yana koy
        imagination_features = torch.cat(imagination_features, dim=1) # [Batch, n_actions * 256]
        
        #  Concatenation
        # İçgüdüler ne diyor + Hayaller ne diyor
        combined_features = torch.cat([mf_features, imagination_features], dim=1)
        
        # Son Karar 
        fc_out = self.fc(combined_features)
        actor_logits = self.actor(fc_out)
        critic_value = self.critic(fc_out)
        
        return actor_logits, critic_value