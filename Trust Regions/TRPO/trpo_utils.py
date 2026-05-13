import torch

def flat_grad(grads):
    """
    PyTorch normalde her katmanın (layer) türevini ayrı bir listede tutar.
    TRPO matematiği matris çarpımları gerektirdiği için, ağın tüm türevlerini
    uc uca ekleyip tek bir devasa 1D (Tek boyutlu) vektör haline getiriyoruz.
    """
    grad_flatten = []
    for grad in grads:
        # view(-1) matrisi tek bir satıra dönüştürür (Flatten)
        grad_flatten.append(grad.view(-1)) 
    return torch.cat(grad_flatten)

def flat_params(model):
    """
    Actor ağımızın o anki tüm ağırlıklarını (parametrelerini) alır
    ve onları yedeklemek/işlem yapmak için tek bir uzun 1D vektör yapar.
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)

def set_params(model, flat_parameters):
    """
    Line Search (Geriye Dönük Arama) yaparken ağın ağırlıklarını sürekli 
    değiştirip denemeler yapacağız. Bu fonksiyon, o düzleştirilmiş (1D) 
    ağırlıkları alır ve tekrar ağın orijinal (2D/3D vb.) katmanlarına geri yerleştirir.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(param.numel())
        # İlgili kısmı kes ve orijinal boyutuna (param.size()) geri çevirerek kopyala
        param.data.copy_(
            flat_parameters[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Yapay Zeka optimizasyonunun kalbi: Eşlenik Gradyan (CG)
    Amacı: Ax = b denklemini, A matrisini (Hessian) RAM'de oluşturmadan çözmek!
    
    f_Ax: Hessian matrisi ile bir vektörün çarpımını veren fonksiyon
    b: Hedef vektör (Bizim durumumuzda Actor'ün ham kaybı)
    cg_iters: Maksimum döngü sayısı (Genelde 10 yeterlidir)
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        # f_Ax fonksiyonu sayesinde Hessian matrisini fiziksel olarak oluşturmadan
        # doğrudan Hessian * P çarpımını (z) elde ediyoruz!
        z = f_Ax(p) 
        
        v = rdotr / (torch.dot(p, z) + 1e-8)
        x += v * p
        r -= v * z
        
        newrdotr = torch.dot(r, r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        
        # Hata payı (residual) çok küçüldüyse erkenden çık (Optimum yön bulundu)
        if rdotr < residual_tol:
            break
            
    return x # Bu x, artık bizim Doğal Gradyanımız (Natural Gradient)