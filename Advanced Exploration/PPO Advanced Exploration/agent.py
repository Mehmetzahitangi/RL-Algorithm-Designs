import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent:
    def __init__(self, network, lr=3e-4, clip_param=0.2, ppo_epochs=4, entropy_coef=0.01, device="cpu"):
        self.network = network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.clip_param = clip_param     # Güvenli Bölge (Trust Region) sınırı
        self.ppo_epochs = ppo_epochs     # Aynı verinin üzerinden kaç kere geçileceği
        self.entropy_coef = entropy_coef # Çok ufak bir entropi (sayım tabanlı olsa da PPO da bunu kullanmak iyidir)
        self.device = device

    def select_action(self, state):
        """Çevreden veriyi al, ağı çalıştır, aksiyonu dön"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_t)
        return action, log_prob.item(), value.item()

    def update(self, rollouts):
        """Toplanan deneyimlerle ağı güvenli bir şekilde güncelle"""
        states, actions, log_probs_old, returns, advantages = rollouts

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Avantajları normalize et (Öğrenmeyi çok dengeler)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Aynı yığını (batch) ppo_epochs kere eğit (PPO'nun veri verimliliği buradan gelir)
        for _ in range(self.ppo_epochs):
            log_probs, entropy, state_values = self.network.evaluate(states, actions)
            state_values = state_values.squeeze()

            # Yeni politikanın eskiye oranı
            ratios = torch.exp(log_probs - log_probs_old)

            # PPO Kesme (Clipping) Matematiği: Çok fazla değişime izin verme!
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Eleştirmen kaybı (Tahmin edilen değer ile gerçek Return arası fark)
            critic_loss = nn.MSELoss()(state_values, returns)

            # Toplam Kayıp: Aktör + Yarım Eleştirmen - Entropi Primi
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return actor_loss.item(), critic_loss.item(), entropy.mean().item()