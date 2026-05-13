import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Öğrenilebilir Parametreler (İkiye ayırıyoruz. Ağın asıl aklı ve gürültü şiddeti)
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        """Ağırlıkları ve gürültü sınırlarını başlat"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Matematiksel optimizasyon: Factorized Gaussian Noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        """
        Bu fonksiyon çağrılana kadar ağ aynı gürültüyle (aynı stratejiyle) oynar.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Dış ve iç gürültüyü çarparak devasa matrisi oluştur
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Eğitimdeysek gürültülü, testteysek saf (mu) ağırlıkları kullan"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)