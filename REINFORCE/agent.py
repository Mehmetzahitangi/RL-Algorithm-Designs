# Ajan sınıfımız (Karar verme ve öğrenme mekanizması)

import torch
import torch.optim as optim
from torch.distributions import Categorical
from policy_network import PolicyNetwork

class ReinforceAgent:

    def __init__(self, obs_size, n_actions, lr=5e-4, gamma=0.99):
        self.gamma = gamma
        # Ajan kendi beynini (ağını) oluşturuyor
        self.policy_net = PolicyNetwork(obs_size, n_actions) # __init__ çağrılır
        # Öğrenme aracı (Optimizer)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        """
        Durumu alır, ağdan olasılıkları çeker, zarı atar ve eylemi döndürür.
        """

        # 1. Veri Hazırlığı: Çoğu simülasyon (Gymnasium vb.) state'i Numpy Array olarak verir
        # PyTorch ise Tensor olarak alır. Girdiyi tensöre çeviriyoruz.
        # unsqueeze(0) ekliyoruz çünkü ağlar veriyi "Batch" (Yığın) halinde bekler. [1, state_boyutu]
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # 2. İleri Besleme: Ağa durumu sor, olasılıkları al (Örn: tensor([[0.XX, 0.yy]]))
        action_probs = self.policy_net(state_tensor) # forward çağrılır

        # 3. Zarı Tasarla: Olasılıklara göre Kategorik bir dağılım oluştur
        m = Categorical(action_probs)

        # 4. Zarı At: Dağılımdan bir eylem örnekle
        # Bu satır %xx ihtimalle 0 (Sol), %yy ihtimalle 1 (Sağ) değerini üretir
        action = m.sample()

        # 5. Logaritmik Olasılık: İleride Loss hesaplamak için bu seçilen eylemin log_prob değerini saklamamız gerekli
        log_prob = m.log_prob(action)
        
        # Ortamın anlayacağı eylemi (sayı olarak) ve eğitimin anlayacağı log_prob'u (tensör olarak) geri dön
        return action.item(), log_prob, m.entropy()


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


    def update_policy(self, log_probs, returns, entropies, beta=0.01):
        """
        Ajanın bölüm (episode) boyunca topladığı verilerle ağı günceller.
        """
        # Reinforce algoritmasındaki Yüksek Varyans sorunu. CartPole'da tüm getiriler +. Bunu Gradient Exploding'e yol açıyor. Standartization ile bir nevi baseline oluştururuz
        # Ortalamanın altında kalanlar - diğerleri + değerlendirilir.
        # 1. Önce normal Python listesini PyTorch Tensörüne çeviriyoruz
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # 2. Z-Score formülünü (x - ortalama) / standart_sapma uyguluyoruz
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9) # + 1e-9 ekledik. Sürekli kötü hamle yapar da 0 gelirse bölme işleminde NaN oluşmasın diye.

        policy_loss = []

        for log_prob, G_t in zip(log_probs, returns_tensor):
            # Eksi ile çarparız (torch içindeki optimizasyon algoritmaları gradient descent için ile lossu minimize etmeye odaklıdır, biz Politika Gradyanı J'yı maksimize etmeye çalışıyoruz bu yüzden - ile çarparız) ve listeye ekle
            policy_loss.append(-log_prob * G_t)

        # PyTorch'un türev alabilmesi için listedeki değerleri toplayıp tek bir Tensör yapıyoruz
        loss = torch.stack(policy_loss).sum()

        total_entropy = torch.stack(entropies).sum() # Keşif Bonusu İçin Ekledik
        loss = loss - (beta * total_entropy) # Loss = Policy Loss - (Beta * Entropy)

        self.optimizer.zero_grad() # eski gradyanları, bir öncekin bölümün hesabını temizleme
        loss.backward() # Gradient hesapla
        self.optimizer.step() # Ağı güncelle