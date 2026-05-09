import gymnasium as gym
import torch
import numpy as np
from network import ActorNetwork
import time

# --- AYARLAR ---
ENV_NAME = "HalfCheetah-v5"
ACTOR_MODEL_PATH = "D4PG_actor_best.pth" # Sadece Actor'ü yüklüyoruz

print("MuJoCo Yükleniyor...")

# 1. Ortamı Görsel (Human) Modda Başlat
env = gym.make(ENV_NAME, render_mode="human")
obs_shape = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

# 2. Donanım ve Ağ Kurulumu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sadece Actor (Oyuncu) ağını başlatıyoruz
actor = ActorNetwork(obs_shape, n_actions).to(device)

# En iyi ağırlıkları yükle
actor.load_state_dict(torch.load(ACTOR_MODEL_PATH, map_location=device))

# Ağı TEST moduna al (Kesinlikle eğitim/türev yok)
actor.eval()

# 3. Simülasyon Döngüsü
state, _ = env.reset()
done = False
total_reward = 0.0

print("Kapatmak için terminalden durdurabilirsiniz.")

while not done:
    # State'i tensöre çevir
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # --- KRİTİK NOKTA: DETERMINİSTİK HAREKET ---
        # Gürültü (Noise) yok! Ağın verdiği hamleyi doğrudan alıyoruz.
        action = actor(state_tensor).cpu().numpy()[0]
        
    # Motor yanmasın diye -1 ile 1 arasına kırp (Güvenlik önlemi)
    action = np.clip(action, -1.0, 1.0)
        
    # Motorlara gücü ver ve sonucu gör
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Döngüyü 0.02 saniye (20 milisaniye) uyut
    # Bu bize saniyede yaklaşık 50 FPS bir görüntü verecek
    time.sleep(0.02)
    
    total_reward += reward
    state = next_state

print(f"Simülasyon Bitti! Gerçekleşen Toplam Skor: {total_reward:.1f}")
env.close()