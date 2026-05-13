import torch
import torch.nn.functional as F
from torch.distributions import Normal
from kfac import KFACOptimizer 

class ACKTRAgent:
    def __init__(self, actor, critic, critic_lr=1e-3, device="cpu"):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        
        # ACTOR İÇİN K-FAC (Doğal Gradyan ve Trust Region burada çalışıyor)
        # kl_clip parametresi, TRPO'daki o meşhur "Güven Bölgesi" sınırımızdır (max_kl).
        self.actor_optimizer = KFACOptimizer(self.actor, lr=0.05, kl_clip=0.01) #lr=0.01
        
        # CRITIC İÇİN STANDART ADAM 
        # Eleştirmenin işi sadece puan tahmin etmektir, o yüzden ona Adam yetiyor.
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        # Ortamda koşturmak için PPO/TRPO ile BİREBİR AYNI zar atma mantığı
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        mu, std = self.actor(state_tensor)
        dist = Normal(mu, std)
        
        action_raw = dist.sample()
        
        log_prob = dist.log_prob(action_raw).sum(dim=-1).detach()
        entropy = dist.entropy().sum(dim=-1).detach()
        value = self.critic(state_tensor).squeeze(-1).detach()
        
        action_clipped = torch.clamp(action_raw, -1.0, 1.0)
        
        return action_clipped.cpu().numpy()[0], action_raw.cpu().numpy()[0], log_prob, entropy, value

    def update(self, rollouts):
        states = rollouts['states'].to(self.device)
        actions = rollouts['actions'].to(self.device)
        returns = rollouts['returns'].to(self.device)
        advantages = rollouts['advantages'].to(self.device)
        
        # CRITIC GÜNCELLEMESİ (Sıradan Regresyon)
        current_values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(current_values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # ACTOR GÜNCELLEMESİ (ACKTR'nin Asıl Olayı Burada)
        # Ağı çalıştır ve mevcut durumlardaki olasılıkları al
        mu, std = self.actor(states)
        dist = Normal(mu, std)
        
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # TRPO'daki o karmaşık oranlara (Ratio) veya PPO'daki kırpmalara (Clipping) GEREK YOK!
        # Çünkü K-FAC optimizer, gradyanı hesaplarken o "Güven Bölgesini" kendi içinde halledecek.
        # Biz sadece "Avantajlı hamlelerin ihtimalini artır" diyen o en ilkel formülü veriyoruz.
        actor_loss = -(new_log_probs * advantages).mean() - 0.01 * entropy

        self.actor_optimizer.zero_grad()
        
        # .backward() dediğimiz an, kfac.py içindeki o gizli kancalar (hooks) çalışacak
        # ve A-B matrislerini saniyeler içinde oluşturacak!
        actor_loss.backward()
        
        # .step() dediğimizde ise Eşlenik Gradyan yerine o matrislerin doğrudan tersi alınacak.
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()