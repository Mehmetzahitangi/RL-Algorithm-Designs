import torch
import torch.nn as nn
import torch.nn.functional as F

class MountainCarNetDistillery(nn.Module):
    def __init__(self, obs_size, hid_size=128):
        super(MountainCarNetDistillery, self).__init__()
        
        # REFERANS AĞ (Daha Derin ve Kaotik)
        # Bu ağ asla eğitilmeyecek. Sadece rastgele ve karmaşık hedefler üretecek.
        self.ref_net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        # Ağın eğitimini tamamen kapatıyoruz (Ağırlıklar dondu)
        self.ref_net.train(False)
        for param in self.ref_net.parameters():
            param.requires_grad = False
            
        # EĞİTİLEN AĞ (Sığ ve Taklitçi)
        # Lapan kitabından bir not: Gözlem uzayımız dar olduğu için bu ağı çok derin yaparsak ezberler (overfitting). Bu yüzden sadece tek katmanlı yapıyoruz
        self.trn_net = nn.Sequential(
            nn.Linear(obs_size, 1),
        )

    def forward(self, x):
        return self.ref_net(x), self.trn_net(x)

    def extra_reward(self, obs):
        """Simülasyonda gezerken o anki durumu tahmin et ve Merak Puanı (Hata) üret"""
        obs_t = torch.FloatTensor([obs])
        with torch.no_grad():
            r1, r2 = self.forward(obs_t)
        # Hata ne kadar büyükse, ajan o kadar "Merak Puanı" kazanır
        return (r1 - r2).abs().detach().numpy()[0][0]

    def loss(self, obs_t):
        """Eğitim sırasında, taklitçi ağı daha iyi tahmin etmesi için güncelle (MSE)"""
        r1_t, r2_t = self.forward(obs_t)
        return F.mse_loss(r2_t, r1_t.detach()).mean()