import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
SeaQuest Taskında çok başarılı olamamıştır
"""
class AtariNetDistillery(nn.Module):
    def __init__(self, input_shape):
        super(AtariNetDistillery, self).__init__()
        
        # 1. REFERANS AĞ (Kaotik ve Dondurulmuş)
        self.ref_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        conv_out_size = self._get_conv_out(self.ref_conv, input_shape)
        
        self.ref_out = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, 1) # Kaotik Hedef
        )
        
        # Referans ağı tamamen dondur!
        self.ref_conv.train(False)
        self.ref_out.train(False)
        for param in self.parameters():
            param.requires_grad = False # Tüm parametreleri dondurduk (Şimdilik)
            
        # 2. EĞİTİLEN AĞ (Taklitçi)
        self.trn_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        
        # Lapan'ın "Ezberlemeyi Önleme" kuralı: Taklitçi ağın sonu sığ olmalı!
        self.trn_out = nn.Sequential(
            nn.Linear(conv_out_size, 1) # Tek katmanlı taklitçi
        )
        
        # Sadece Eğitilen Ağı (trn_conv ve trn_out) çözüyoruz
        for param in self.trn_conv.parameters():
            param.requires_grad = True
        for param in self.trn_out.parameters():
            param.requires_grad = True

    def _get_conv_out(self, conv_net, shape):
        o = conv_net(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Referans Ağın Çıktısı
        ref_features = self.ref_conv(x).view(x.size()[0], -1)
        ref_val = self.ref_out(ref_features)
        
        # Eğitilen Ağın Çıktısı
        trn_features = self.trn_conv(x).view(x.size()[0], -1)
        trn_val = self.trn_out(trn_features)
        
        return ref_val, trn_val

    def extra_reward(self, obs_t):
        """O anki piksellere bak, tahmin hatasını Merak Puanı olarak dön."""
        with torch.no_grad():
            r1, r2 = self.forward(obs_t)
        return (r1 - r2).abs().detach().cpu().numpy()[0][0]

    def loss(self, obs_t):
        """Eğitim sırasında taklitçi ağı referansa yaklaştır."""
        r1_t, r2_t = self.forward(obs_t)
        return F.mse_loss(r2_t, r1_t.detach()).mean()