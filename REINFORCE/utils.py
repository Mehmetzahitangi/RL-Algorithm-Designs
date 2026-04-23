import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(rewards, window=20):
    """
    Eğitim sürecindeki ödülleri ve hareketli ortalamayı çizer.
    """

    plt.figure(figsize=(10,5))

    # 1. Ham ödüller (Arka plan gürültüsü)
    plt.plot(rewards, color='lightskyblue', alpha=0.6, label='Bölüm Ödülü')

    # 2. Hareketli ortalama
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        # X eksenini kaydırmamak için baştaki boşlukları None veya 0 ile dolduruyoruz
        # Basitlik için moving_avg'yi doğrudan window'dan başlatarak çiziyoruz
        plt.plot(range(window-1, len(rewards)), moving_avg, color='blue', linewidth=2, label=f'{window} Bölümlük Ortalama')

    plt.title('Ajanın Öğrenme Eğrisi (REINFORCE)')
    plt.xlabel('Bölüm (Episode)')
    plt.ylabel('Toplam Ödül')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Grafiği ekranda göster ve istersen kaydet
    plt.savefig('learning_curve.png')
    plt.show()