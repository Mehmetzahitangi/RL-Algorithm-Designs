import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvironmentModel(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(EnvironmentModel, self).__init__()
        self.n_actions = n_actions
        
        # 1. Girdi İşleme (Observation ve Aksiyonu Birleştirme)
        # DÜZELTME: obs_shape[0] + 1 YERİNE obs_shape[0] + n_actions 
        # Yani 4 karelik görüntüye, 4 tane de aksiyon katmanı ekliyoruz (Toplam 8 kanal)
        self.conv1 = nn.Conv2d(obs_shape[0] + n_actions, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # 2. Görüntüyü Yeniden Oluşturma (Deconvolution / Transposed Conv)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, obs_shape[0], kernel_size=4, stride=2, padding=1)
        
        # 3. Ödül Tahmincisi (Reward Head)
        self.reward_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        # --- KRİTİK DÜZELTME: 64 * 4 * 4 YERİNE 5184 YAZIYORUZ ---
        self.reward_fc = nn.Sequential(
            nn.Linear(5184, 128), # 64 kanal * 9 * 9 uzaysal boyut = 5184
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, action_one_hot):
        batch_size = obs.size()[0]
        
        # --- KRİTİK DÜZELTME BURADA ---
        # action_one_hot boyutu: [Batch, 4]
        # Bunu [Batch, 4, Yükseklik, Genişlik] formatına esnetiyoruz!
        action_plane = action_one_hot.view(batch_size, self.n_actions, 1, 1).expand(
            batch_size, self.n_actions, obs.size()[2], obs.size()[3]
        )
        
        # Görüntü (4 kanal) ve Aksiyon (4 kanal) birleşiyor -> Toplam 8 kanal
        x = torch.cat([obs, action_plane], dim=1)
        
        # Çevre modelinin algısı
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Delta Görüntü Tahmini
        out_obs_delta = F.relu(self.deconv1(x))
        out_obs_delta = self.deconv2(out_obs_delta)
        
        next_obs_prediction = obs + out_obs_delta
        
        # Ödül Tahmini
        r = self.reward_conv(x)
        r = r.view(batch_size, -1)
        reward_prediction = self.reward_fc(r)
        
        return next_obs_prediction, reward_prediction