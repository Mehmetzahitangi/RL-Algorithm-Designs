import gymnasium as gym
import torch
import numpy as np
from atari_wrappers import make_env
from network import ActorCriticNetwork

# Ayarlar
ENV_NAME = "ALE/Pong-v5"
MODEL_PATH = "a2c_pong_best.pth" 

# Ortamı İnsan Gözü İçin Başlatıyoruz (Vektörize değil, Tekil)
print("Ortam yükleniyor...")
env = make_env(ENV_NAME, render_mode="human")
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCriticNetwork(obs_shape, n_actions).to(device)

# Ağırlıkları yükle ve Test moduna al
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

state, _ = env.reset()
done = False
total_reward = 0.0

print("Oyun başlıyor...")

while not done:
    # Kritik: Gelen (4, 84, 84) resmini (1, 4, 84, 84) yapıp GPU'ya yolluyoruz
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        action_probs, _ = model(state_tensor)
        
        # Eğitimde "Categorical" ile zar atıyorduk (Keşif için).
        # Testte ise ağın "en emin olduğu" eylemi doğrudan seçiyoruz (Açgözlü / Greedy)
        action = torch.argmax(action_probs, dim=1).item()
        
    # Ortamda 1 adım
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    total_reward += reward
    state = next_state

print(f"Oyun Bitti! Toplam Skor: {total_reward}")
env.close()