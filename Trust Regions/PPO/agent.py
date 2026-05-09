import torch
import torch.nn.functional as F
from torch.distributions import Normal

class PPOAgent:
    def __init__(self, actor, critic, actor_lr=3e-4, critic_lr=3e-4, device="cpu", clip_param=0.2, ppo_epochs=10):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.device = device
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs

    def select_action(self, state):
            # 1. NumPy dizisini PyTorch Tensörüne çevir ve Batch boyutu (unsqueeze) ekle!
            # Boyut (17,) iken (1, 17) olacak. Bu, ağın çökmesini engeller.
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 2. Actor ağından Çan Eğrisinin parametrelerini al
            mu, std = self.actor(state_tensor)
            
            dist = Normal(mu, std)

            # HAM AKSİYON (Bunu hafızaya kaydedeceğiz)
            action_raw = dist.sample()
            
            # Matematiksel Güvenlik: Detach kullanarak hesaplama grafiğini koparıyoruz
            log_prob = dist.log_prob(action_raw).sum(dim=-1).detach()
            entropy = dist.entropy().sum(dim=-1).detach()
            value = self.critic(state_tensor).squeeze(-1).detach()
            
            action_clipped = torch.clamp(action_raw, -1.0, 1.0)
            
            return action_clipped.cpu().numpy()[0], action_raw.cpu().numpy()[0], log_prob, entropy, value




    def update(self, rollouts, batch_size=64):
            
            states = rollouts['states'].to(self.device)
            actions = rollouts['actions'].to(self.device)
            returns = rollouts['returns'].to(self.device)
            advantages = rollouts['advantages'].to(self.device)


            # ÖNEMLİ: Eski log_prob'ları "detach" yapıyoruz. 
            # Onlar artık eğitilecek bir şey değil, sadece birer referans noktası (sabit sayı).
            old_log_probs = rollouts['log_probs'].to(self.device).detach()

            dataset_size = states.size(0)

            # PPO'nun Gücü: Aynı veriyi 1 kez değil, 10 kez (Epoch) kullanarak eğitiriz.
            # Clipping (Kırpma) sayesinde ağı bozmadan aynı veriyi defalarca sağabiliriz.
            for _ in range(self.ppo_epochs):
                # Verileri her epoch'ta karıştır (Ezberi boz)
                indices = torch.randperm(dataset_size)

                # Veriyi 64'lük mini-batchlere böl
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    mb_indices = indices[start:end]

                    # O anki Mini-Batchi çek
                    mb_states = states[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                     
                    mu, std = self.actor(mb_states)
                    dist = Normal(mu, std)

                    # Yeni politikaya göre bu hareketlerin YAPILMA İHTİMALİ
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    current_values = self.critic(mb_states).squeeze(-1)

                    # ORAN (RATIO) HESAPLAMASI (Yeni / Eski)
                    # Logaritma kuralı: exp(Yeni_Log - Eski_Log) bize doğrudan oranı verir
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    # SURROGATE (VEKİL) LOSS VE KIRPMA (CLIPPING) İŞLEMİ
                    # Senaryo 1: Kırpmasız normal çarpım (Eski A2C'nin oranlı hali)
                    surr1 = ratio * mb_advantages
                    
                    # Senaryo 2: Oranı [0.8, 1.2] arasına zorla (Kırp)
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                    # PPO Kuralı: İki senaryodan HANGİSİ DAHA KÖTÜYSE onu seç (min).
                    # PyTorch küçültmeye çalıştığı için en son eksi (-) ile çarpıyoruz.
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropi bonusu (Ajan keşfetmeye devam etsin)
                    actor_loss = actor_loss - 0.01 * entropy

                    # CRITIC (ELEŞTİRMEN) LOSS
                    critic_loss = F.mse_loss(current_values, mb_returns)

                    # 5. GERİYE YAYILIM (BACKPROP) - ACTOR
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    # 6. GERİYE YAYILIM (BACKPROP) - CRITIC
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()

            return critic_loss.item(), actor_loss.item()