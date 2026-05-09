import torch
import torch.nn.functional as F
from torch.distributions import Normal

class A2CAgent:
    def __init__(self, actor, critic, actor_lr=3e-4, critic_lr=3e-4, device="cpu"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        # PPO ve modern A2C mimarilerinde genelde Adam optimizer kullanılır
        # ve her iki ağ için de öğrenme oranları nispeten küçük/aynı tutulur
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.device = device

    def select_action(self, state):
            # 1. NumPy dizisini PyTorch Tensörüne çevir ve Batch boyutu (unsqueeze) ekle!
            # Boyut (17,) iken (1, 17) olacak. Bu, ağın çökmesini engeller.
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 2. Actor ağından Çan Eğrisinin parametrelerini al
            mu, std = self.actor(state_tensor)
            
            # 3. Normal Dağılım oluştur ve örneklem (sample) çek
            dist = Normal(mu, std)
            action = dist.sample()
            
            # 4. Log-Prob ve Entropi (Boyutlar (1, 6) olduğu için sum son eksende çalışır ve (1,) döner)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # 5. Critic'ten o anki durumun Değerini (V(s)) al
            # Critic (1, 1) döner, squeeze(-1) ile (1,) yaparız
            value = self.critic(state_tensor).squeeze(-1)
            
            # 6. Motor yanmasın diye kırp
            action_clipped = torch.clamp(action, -1.0, 1.0)
            
            # Çevreye göndermek için Eylemi tekrar NumPy'a çevir ve Batch boyutundan [0] ile çıkart
            # Log-prob, entropy ve value değerlerini tensör olarak bırakıyoruz (main.py'de torch.cat ile birleşecekler)
            return action_clipped.cpu().numpy()[0], log_prob, entropy, value


    def calculate_returns(self, rewards):
        """
        Ödül listesini alır ve indirgenmiş getirileri (G_t) hesaplar.
        """
        returns = []
        G_next = 0.0  # En son adımdan sonra bir oyun olmadığı için gelecekteki getiri 0'dır

        for r in reversed(rewards):
            G_current = r + (self.gamma * G_next)

            returns.append(G_current)

            G_next = G_current

        returns.reverse()

        return returns


    def update(self, rollouts):
            """
            rollouts dict şunları içermelidir (hepsi tensör olacak şekilde):
            states, actions, log_probs, returns, values, advantages, entropies
            """
            # Verileri kolay kullanım için değişkenlere al
            states = rollouts['states'].to(self.device)
            actions = rollouts['actions'].to(self.device)
            old_log_probs = rollouts['log_probs'].to(self.device)
            returns = rollouts['returns'].to(self.device)   # Gelecek ödüllerin hesaplanmış hali (G_t)
            values = rollouts['values'].to(self.device)     # Critic'in daha önce tahmin ettiği V(s)
            advantages = rollouts['advantages'].to(self.device) # Gerçekleşen - Beklenen
            old_entropies = rollouts['entropies'].to(self.device)


            # Critic Loss: Tahmin Hatasını (MSE) Küçült

            # Critic ağından o anki durumların güncel değerlerini tekrar al
            current_values = self.critic(states).squeeze(-1)
            
            # Ortalama Kare Hatası (MSE Loss)
            critic_loss = F.mse_loss(current_values, returns)
            
            # Critic'i güncelle
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Gradient Clipping: Critic çok büyük bir hata yapıp ağı patlatmasın diye türevleri sınırla
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()


            # Actor Loss: Başarılı Hamlelerin İhtimalini Artır
            # Actor'ün kaybı, daha önce hesapladığımız (ve rollouts'tan gelen) log_prob'lar ile
            # Avantajların çarpımından oluşur. (Eksi işareti gradyan yükseltmesi (ascend) yapmak içindir)
            
            # A2C için klasik formül (LogProb * Advantage)
            actor_loss = -(old_log_probs * advantages).mean()
            
            # Entropi Bonusu: Çok emin olmayı (çan eğrisini daraltmayı) cezalandırır
            # Ajanı keşfetmeye teşvik eder. (0.01 entropi katsayısı standarttır)
            entropy_loss = old_entropies.mean()
            
            # Toplam Actor Kaybı (Entropi bonusunu çıkartıyoruz çünkü PyTorch kayıpları küçültür)
            total_actor_loss = actor_loss - 0.01 * entropy_loss

            # Actor'ü güncelle
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            # Gradient Clipping: Actor de aniden çok büyük bir adım atmasın
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            return critic_loss.item(), total_actor_loss.item()