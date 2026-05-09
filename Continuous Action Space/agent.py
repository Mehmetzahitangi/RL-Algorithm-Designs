import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from network import ActorNetwork, DistributionalCriticNetwork

class OUNoise:
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class DDPGAgent:
    # Ajan artık tau (klonların güncellenme hızı) diye bir hiperparametreye sahip
    def __init__(self, obs_shape, n_actions, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau 
        self.n_actions = n_actions

        # --- C51: YENİ HİSTOGRAM AYARLARI ---
        self.v_min = -1000.0
        self.v_max = 2000.0
        self.n_atoms = 51
        self.dz = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # X Ekseni: -1000'den 2000'e kadar 51 eşit parça (Tensör olarak GPU'da duracak)
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        
        # 1. ASIL AĞLAR (Sürekli eğitilen, hızlı değişen)
        self.actor = ActorNetwork(obs_shape, n_actions).to(self.device)
        self.critic = DistributionalCriticNetwork(obs_shape, n_actions, self.v_min, self.v_max, self.n_atoms).to(self.device)
        
        # 2. HEDEF AĞLAR (Asıl ağların dondurulmuş klonları)
        self.target_actor = ActorNetwork(obs_shape, n_actions).to(self.device)
        self.target_critic = DistributionalCriticNetwork(obs_shape, n_actions, self.v_min, self.v_max, self.n_atoms).to(self.device)
        
        # Başlangıçta Klonların ağırlıklarını asıl ağlarla birebir aynı yapıyoruz
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizerlar (Sadece Asıl Ağlar eğitilir Klonlar türevle eğitilmez)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Gürültü
        self.noise = OUNoise(n_actions)
        
    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Ağı test moduna alıp kesin kararımızı (mu) çekiyoruz
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        # Eğitimdeysek o rüzgarı (gürültüyü) hareketin üstüne ekliyoruz
        if add_noise:
            action += self.noise.sample()
            
        # Motor yanmasın diye gelen sinyali kesin olarak -1 ile 1 arasına kırpıyoruz (Clipping)
        return np.clip(action, -1.0, 1.0)
    

    def update(self, replay_buffer, batch_size=64):
        # Eğer hafızada yeterli veri yoksa eğitimi başlatma
        if len(replay_buffer) < batch_size:
            return None, None
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # --- CRITIC GÜNCELLEMESİ (CATEGORICAL PROJECTION) --- 
        # Klon ağları kullanarak bir adım sonrasının hareketini ve değerini tahmin et
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_probs = self.target_critic(next_state, next_action) # Sonuç: (Batch, 51) Olasılıklar

            # Gelecekteki 51 çubuğun Puan değerlerini (X ekseni) Bellman ile sağa/sola kaydır
            # Formül: Tz = R + gamma * Z
            Tz = reward + (1 - done) * self.gamma * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max) # Sınırların dışına çıkmayı yasakla
            
            # --- PROJEKSİYON BAŞLANGICI ---
            # Kaymış puanların, bizim sabit 51 slotun hangilerine denk geldiğini bul
            b = (Tz - self.v_min) / self.dz
            l = b.floor().clamp(0, self.n_atoms - 1) # Sol komşu yuva
            u = b.ceil().clamp(0, self.n_atoms - 1)  # Sağ komşu yuva

            # Hangi slota ne kadar olasılık verileceğini hesapla
            dl = u - b
            du = b - l
            dl[(l == u)] = 1.0 # Tam üstüne denk geldiyse kaybolmayı önle
            du[(l == u)] = 0.0

            # Yeni hedef (target) grafiği sıfırdan oluştur ve olasılıkları slotlara koy
            target_probs = torch.zeros_like(next_probs)
            for i in range(batch_size):
                target_probs[i].index_add_(0, l[i].long(), next_probs[i] * dl[i])
                target_probs[i].index_add_(0, u[i].long(), next_probs[i] * du[i])

        # Asıl Critic ağımızın o anki tahmini
        current_probs = self.critic(state, action)

        # CRITIC LOSS: Cross-Entropy (İki grafiğin birbiriyle ne kadar eşleştiği)
        # Sıfır logaritma hatası vermesin diye 1e-6 ekliyoruz
        critic_loss = -torch.sum(target_probs * torch.log(current_probs + 1e-6), dim=1).mean()

        # Critic'in backprop ile eğitimi
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- ACTOR AĞININ  GÜNCELLENMESİ (POLICY GRADIENT) ---

        # Actor'ün şu anki duruma göre yapmak istediği hamleler
        actor_action = self.actor(state)
        actor_probs = self.critic(state, actor_action)
        
        # Actor "Beklenen Değeri" (Expected Value) maksimize etmeye çalışır
        # Beklenen Değer = Olasılıklar x X_Ekseni_Puanları
        expected_Q = torch.sum(actor_probs * self.support, dim=1)

        # PyTorch/Tensoflow küçültmeye çalıştığı için (-) ile çarpıyoruz
        actor_loss = -expected_Q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- HEDEF AĞLARIN YUMUŞAK GÜNCELLENMESİ (Polyak Averaging) ---
        # Klonları asıl ağlara doğru %0.5 (tau) oranında çok yavaşça yaklaştırıyoruz
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return critic_loss.item(), actor_loss.item()