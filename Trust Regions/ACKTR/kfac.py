import torch
import torch.optim as optim

class KFACOptimizer(optim.Optimizer):
    """
    TRPO'nun yavaşlığını çözen, Fisher Bilgi Matrisini Kronecker Çarpımı 
    ile sıkıştıran efsanevi ACKTR motoru.
    """
    def __init__(self, model, lr=0.25, kl_clip=0.001, weight_decay=0, stat_decay=0.99):
        defaults = dict(lr=lr, kl_clip=kl_clip, weight_decay=weight_decay)
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        
        self.kl_clip = kl_clip
        self.stat_decay = stat_decay
        self.model = model
        self.modules = []
        self.steps = 0
        
        # Sadece Linear katmanlara K-FAC kancası (hook) atıyoruz
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                self.modules.append(module)
                self._register_hooks(module)

        self.state = {module: {'A': None, 'G': None} for module in self.modules}

    def _register_hooks(self, module):
        """Ağın içine sızıp İleri ve Geri akışları dinliyoruz"""
        def forward_hook(mod, input, output):
            # İleri Akış: Girdileri (Aktivasyonları) yakala
            a = input[0].data
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1) # Bias ekle
            self.state[mod]['a'] = a

        def backward_hook(mod, grad_input, grad_output):
            # Geri Akış: Çıktı gradyanlarını yakala
            g = grad_output[0].data
            self.state[mod]['g'] = g

        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

    def step(self):
        """Adam optimizer'ın step() fonksiyonunun K-FAC versiyonu"""
        self.steps += 1
        vg_sum = 0
        
        # 1. Her katman için Fisher matrisinin K-FAC sıkıştırılmış hallerini (A ve G) hesapla
        for module in self.modules:
            a = self.state[module]['a']
            g = self.state[module]['g']
            
            # Kronecker bileşenleri (A ve G matrisleri)
            A = torch.matmul(a.t(), a) / a.size(0)
            G = torch.matmul(g.t(), g) / g.size(0)
            
            # Hareketli ortalama (Moving average) ile istikrar sağla
            if self.state[module]['A'] is None:
                self.state[module]['A'] = A
                self.state[module]['G'] = G
            else:
                self.state[module]['A'] = self.state[module]['A'] * self.stat_decay + A * (1 - self.stat_decay)
                self.state[module]['G'] = self.state[module]['G'] * self.stat_decay + G * (1 - self.stat_decay)

            # 2. Matrislerin tersini al (Küçük oldukları için anında çözülür!)
            damp = 1e-2 # Sıfıra bölmeyi engellemek için güvenlik payı
            inv_A = torch.linalg.inv(self.state[module]['A'] + damp * torch.eye(A.size(0), device=A.device))
            inv_G = torch.linalg.inv(self.state[module]['G'] + damp * torch.eye(G.size(0), device=G.device))
            
            # 3. Doğal Gradyanı (Natural Gradient) hesapla
            grad = module.weight.grad.data
            bias_grad = module.bias.grad.data
            grad_matrix = torch.cat([grad, bias_grad.unsqueeze(1)], 1)
            
            # K-FAC'ın Sihri: inv_G * gradient_matrix * inv_A
            v = torch.matmul(inv_G, torch.matmul(grad_matrix, inv_A))
            self.state[module]['v'] = v
            
            # KL Diverjans sınırını kontrol etmek için vektör-gradyan çarpımını topla
            vg_sum += (v * grad_matrix).sum()

        # 4. TRPO'nun o meşhur Güven Bölgesi (Trust Region) oranını ayarla
        nu = min(1.0, torch.sqrt(self.kl_clip / (vg_sum + 1e-8)))

        # 5. Ağırlıkları Güvenli Adımla (Natural Gradient) Güncelle
        for module in self.modules:
            v = self.state[module]['v']
            weight_update = v[:, :-1]
            bias_update = v[:, -1]
            
            module.weight.data -= self.param_groups[0]['lr'] * nu * weight_update
            module.bias.data -= self.param_groups[0]['lr'] * nu * bias_update