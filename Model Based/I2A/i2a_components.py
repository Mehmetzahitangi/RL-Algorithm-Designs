import torch
import torch.nn as nn
import torch.nn.functional as F

# HAYAL İÇİ PİLOT (Rollout Policy) 
class RolloutPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(RolloutPolicy, self).__init__()
        # Ana beynin aksine çok hafif bir CNN kullanıyoruz ki hayaller hızlı aksın
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 84x84 Atari pikselleri için hesaplanmış standart çıkış boyutu
        conv_out_size = 3136 
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        fx = self.conv(x).view(x.size()[0], -1)
        return self.fc(fx) # Logits döner (Aksiyon ihtimalleri)


#  HAYAL KODLAYICI (Rollout Encoder) 
class RolloutEncoder(nn.Module):
    def __init__(self, obs_shape, hidden_size=256):
        super(RolloutEncoder, self).__init__()
        
        # Hayal edilen ekranları anlamlandırmak için CNN
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = 3136
        
        # Görüntü özelliklerini (features) ve o adımdaki hayali ödülü birleştiriyoruz
        # Bu yüzden +1 (ödül için) ekleniyor.
        self.rnn = nn.GRU(input_size=conv_out_size + 1, hidden_size=hidden_size, batch_first=True)

    def forward(self, obs_seq, reward_seq):
        """
        obs_seq: [Batch, Adım_Sayısı, Kanal, Yükseklik, Genişlik]
        reward_seq: [Batch, Adım_Sayısı, 1]
        """
        batch_size, seq_len, c, h, w = obs_seq.size()
        
        # RNN'e vermeden önce tüm adımlardaki görüntüleri tek seferde CNN'den geçir (Hız optimizasyonu)
        obs_seq_flat = obs_seq.view(batch_size * seq_len, c, h, w)
        conv_out = self.conv(obs_seq_flat)
        conv_out = conv_out.view(batch_size, seq_len, -1)
        
        # CNN'den çıkan görüntü özellikleri ile hayali ödülleri uç uca ekle
        rnn_input = torch.cat([conv_out, reward_seq], dim=-1)
        
        # Zaman Serisi Ağı (GRU) çalışıyor... Bize sadece en son adımın (özetin) çıktısı lazım!
        _, hidden_state = self.rnn(rnn_input)
        
        # hidden_state boyutu: [1, Batch, Hidden_Size] -> Bunu düzleştirip [Batch, Hidden_Size] yapıyoruz
        return hidden_state.squeeze(0)