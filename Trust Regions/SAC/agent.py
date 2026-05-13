import torch
import torch.nn.functional as F
import copy
import numpy as np

class SACAgent:
    def __init__(self, actor, critic, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, 
                 gamma=0.99, tau=0.005, device="cpu", action_dim=6):
        
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # TEMEL AĞLAR
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        # HEDEF AĞ (Sadece Critic için, hedefin sürekli değişmesini engeller)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        
        # OTOMATİK ENTROPİ (Sıcaklık - Alpha) AYARI
        # Hedef Entropi genelde eksi Aksiyon Boyutudur (Örn: HalfCheetah için -6.0)
        self.target_entropy = -float(action_dim)
        # Alpha'yı bir tensör yapıyoruz ki PyTorch bunun türevini alabilsin!
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item() # Başlangıçta exp(0) = 1.0

        # 4. OPTİMİZASYON MOTORLARI (Adam)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, state, evaluate=False):
        """Ajan ortamda koştururken (Evaluate=False) veya Test edilirken (Evaluate=True)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # Test sırasında zar atma, net bildiğini (mu) yap
                _, _, action = self.actor.sample(state_tensor)
            else:
                # Eğitim sırasında zar at ve Reparameterization ile kaotik davran
                action, _, _ = self.actor.sample(state_tensor)
                
        return action.cpu().numpy()[0]

    def update(self, batch):
        """Hafıza havuzundan (Replay Buffer) gelen 256'lık veri yığını ile eğitimi başlat"""
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)


        # 1. CRITIC GÜNCELLEMESİ (İkiz Eleştirmen Matematiği)
        with torch.no_grad():
            # Gelecekteki durumu Actor'e sor, bize aksiyon ve olasılığını versin
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            
            # Gelecek için İKİ eleştirmenden de puan iste
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            
            # Kötümserlik Kuralı: Hangisi düşük puan verdiyse onu seç (Overestimation'ı bitirir)
            min_target_q = torch.min(target_q1, target_q2)
            
            # Hedef Q Değeri: Puan + (Gelecekteki Değer - Alpha * Gelecekteki Kaos)
            target_q = rewards + (1 - dones) * self.gamma * (min_target_q - self.alpha * next_log_pi)

        # O anki durum için Q değerlerini hesapla
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic Kaybı (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. ACTOR GÜNCELLEMESİ (Reparameterization Trick Devrede)
        # Actor'den O ANKİ durumlar için yepyeni aksiyonlar iste (Türevler açık)
        curr_actions, curr_log_pi, _ = self.actor.sample(states)
        
        # Bu yeni uydurulan hareketler kaç puan eder?
        q1_new, q2_new = self.critic(states, curr_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Actor'ün Hedefi: Ödülü Maksimize Et (Eksi koyuyoruz çünkü optimizer minimize eder) 
        # VE Entropiyi (Kaosu) Maksimize et
        actor_loss = (self.alpha * curr_log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ALPHA (SICAKLIK) GÜNCELLEMESİ
        # Ağ kendi kendine karar veriyor: Entropi sınırının altında mıyız, üstünde miyiz kontrolü
        alpha_loss = -(self.log_alpha * (curr_log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Alpha'yı güncel değere çek
        self.alpha = self.log_alpha.exp().item()

        # HEDEF AĞLARI YUMUŞAK GÜNCELLEME (Polyak Averaging)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return critic_loss.item(), actor_loss.item(), self.alpha