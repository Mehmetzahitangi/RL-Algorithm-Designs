import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from network import Network

class ReinforceAgent:

    def __init__(self, obs_size, n_actions, lr=7e-4, gamma=0.999):
        self.gamma = gamma
        # Ajan kendi beynini (ağını) oluşturuyor
        self.policy_net = Network(obs_size, n_actions) # __init__ çağrılır
        # Öğrenme aracı (Optimizer)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):

        #state zaten (16, 8) boyutunda. unsqueeze(0) kullanmıyoruz
        state_tensor = torch.from_numpy(state).float() 
        
        # Ağ artık 16 ajan için 16 farklı olasılık dağılımı ve 16 durum değeri üretiyor
        action_probs, state_value = self.policy_net(state_tensor)  # Actorden action_probs, Criticden state_value

        m = Categorical(action_probs)
        action = m.sample() # action boyutu artık [16] önceden tensor(5) gibi tek bir değerdi, action.item() ile integer'a dönüştürüyorduk
        
        # Ortama göndermek için 16'lık Tensörü, 16'lık Numpy dizisine çeviriyoruz
        action_array = action.cpu().numpy()

        return action_array, m.log_prob(action), m.entropy(), state_value

    def update_policy(self, log_probs, values, returns, entropies, beta=0.01):

        returns_tensor = torch.stack(returns) # Normalize etmeyiz. Critic'in gerçek puanı Gt tahmin etmesini isteriz. Return'ler tensor olarak geliyor artık, 128 tensor için stack kullanmalıyız
       
        # log_probs, values ve entropies zaten tensör,  sadece alt alta yığıyoruz.
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values).squeeze(-1) # (128, 16, 1) boyutunu (128, 16) yapmak için
        entropies_tensor = torch.stack(entropies)
        
        
        # Advantage hesabı (Critic'in türevini kopararak)
        advantages = returns_tensor - values_tensor.detach()
        
        # REINFORCE'un aksine getirileri/return değil avantajları normalize ederiz
        # Avantajları normalize ederek Actor'ün eğitimini stabil hale getiriyoruz. Teoride şart değil ama sinir ağları her zaman [-1, 1] arasında değişen, standart sapması 1 olan verilerle çok daha istikrarlı öğrenir
        # Avantajlar bazen +50 bazen -2 çıkabilir. Bu dalgalanma ağın öğrenme oranını (learning rate) dengesizleştirir.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9) 

        actor_loss = -(log_probs_tensor * advantages).sum()

        critic_loss = F.mse_loss(values_tensor, returns_tensor)

        entropy_loss = entropies_tensor.sum()

        loss = actor_loss + critic_loss - (beta * entropy_loss)

        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item() # TensorBoard için


    def calculate_returns(self, rewards, masks, next_values):
        returns = []
        G_next = next_values.squeeze(-1) # next_values 16 ajanın her biri için 1 değerdir, .squeeze(-1) ile boyutu (16,1) den (16,) e çeviriyoruz

        for r, mask in zip(reversed(rewards), reversed(masks)):
            r_tensor = torch.tensor(r, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)

            G_current = r_tensor + (self.gamma * G_next * mask_tensor)  
            

            returns.append(G_current)

            G_next = G_current

        returns.reverse()

        return returns