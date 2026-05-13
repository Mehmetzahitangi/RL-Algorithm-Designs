import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from trpo_utils import flat_grad, flat_params, set_params, conjugate_gradient

class TRPOAgent:
    def __init__(self, actor, critic, critic_lr=1e-3, device="cpu", max_kl=0.01, cg_damping=0.1):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        # Sadece Critic için optimizer var! 
        # Actor'ü manuel olarak, matematikle güncelleyeceğiz
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.max_kl = max_kl          # Güven Bölgesinin yarıçapı (TRPO'nun altın kuralı)
        self.cg_damping = cg_damping  # Hessian hesaplamasındaki sayısal dengeleyici
        self.device = device

    def select_action(self, state):
        # Burası PPO ile tamamen aynı, ajan dünyayı keşfediyor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, std = self.actor(state_tensor)
        dist = Normal(mu, std)
        
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1).detach()
        value = self.critic(state_tensor).squeeze(-1).detach()
        
        action_clipped = torch.clamp(action_raw, -1.0, 1.0)
        return action_clipped.cpu().numpy()[0], action_raw.cpu().numpy()[0], log_prob, value

    def hessian_vector_product(self, states, old_dist, vector):
        """Hessian matrisini oluşturmadan H*v çarpımını bulan matematiksel sihir"""
        mu, std = self.actor(states)
        new_dist = Normal(mu, std)
        
        # 1. KL Diverjansını hesapla
        kl = kl_divergence(old_dist, new_dist).sum(dim=-1).mean()
        
        # 2. Birinci Türev (Gradyan) (create_graph=True çünkü bunun da türevini alacağız!)
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_kl_grad = flat_grad(kl_grad)
        
        # 3. Gradyan ile Vektörü Çarp (Dot Product)
        kl_v = torch.dot(flat_kl_grad, vector)
        
        # 4. İkinci Türevi al (Hessian-Vector Product doğuyor!)
        kl_v_grad = torch.autograd.grad(kl_v, self.actor.parameters())
        flat_kl_v_grad = flat_grad(kl_v_grad)
        
        return flat_kl_v_grad + vector * self.cg_damping

    def update(self, rollouts):
        states = rollouts['states'].to(self.device)
        actions = rollouts['actions'].to(self.device)
        returns = rollouts['returns'].to(self.device)
        advantages = rollouts['advantages'].to(self.device)
        old_log_probs = rollouts['log_probs'].to(self.device).detach()

        # 1. CRITIC GÜNCELLEMESİ (Sıradan Regresyon)
        # Eleştirmenin işi sadece puan tahmin etmektir, Adam yeterli.
        for _ in range(10): # Critic'i 10 kez üst üste eğitiyoruz
            current_values = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(current_values, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


        # 2. ACTOR GÜNCELLEMESİ (TRPO'nun Kalbi)
        # A) Referans (Eski) Dağılımı Dondur
        with torch.no_grad():
            old_mu, old_std = self.actor(states)
            old_dist = Normal(old_mu, old_std)

        # B) Surrogate Loss (Vekil Kayıp) Hesapla (PPO'daki kırpmasız oran)
        mu, std = self.actor(states)
        dist = Normal(mu, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate_loss = (ratio * advantages).mean()

        # C) Kaybın Birinci Türevini Al
        loss_grad = torch.autograd.grad(surrogate_loss, self.actor.parameters())
        loss_grad_flat = flat_grad(loss_grad)

        # D) Eşlenik Gradyan (Conjugate Gradient) ile Kusursuz Yönü Bul
        def f_Ax(v):
            return self.hessian_vector_product(states, old_dist, v)
        
        step_dir = conjugate_gradient(f_Ax, loss_grad_flat)

        # E) Adım Büyüklüğünü Hesaplan (Lagrange Çarpanları Matematiği)
        shs = 0.5 * torch.dot(step_dir, f_Ax(step_dir))
        lm = torch.sqrt(shs / self.max_kl)
        full_step = step_dir / lm

        # F) LINE SEARCH (Geriye Dönük Arama)
        old_params = flat_params(self.actor)
        success = False
        step_size = 1.0 # Adıma tam boy (1.0) ile başla
        
        for _ in range(10): # 10 kez küçülterek dene
            new_params = old_params + step_size * full_step
            set_params(self.actor, new_params) # Ağa yeni ağırlıkları yükle
            
            # Yeni ağın performansını test et
            with torch.no_grad():
                mu_new, std_new = self.actor(states)
                dist_new = Normal(mu_new, std_new)
                
                new_log_probs_test = dist_new.log_prob(actions).sum(dim=-1)
                ratio_test = torch.exp(new_log_probs_test - old_log_probs)
                new_surrogate_loss = (ratio_test * advantages).mean()
                
                # En kritik test: KL sınırını aştık mı?
                kl_test = kl_divergence(old_dist, dist_new).sum(dim=-1).mean()
            
            # Eğer yeni politika daha iyiyse VE Güven Bölgesi (max_kl) aşılmadıysa: Kabul et!
            if new_surrogate_loss > surrogate_loss and kl_test <= self.max_kl:
                success = True
                break
            
            # Olmadıysa, adımı yarıya indir ve geriye dön
            step_size *= 0.5
            
        # Eğer 10 denemede de sınırın içine giremediysek, ağırlıkları eski haline getir.
        if not success:
            set_params(self.actor, old_params)

        return critic_loss.item(), surrogate_loss.item()