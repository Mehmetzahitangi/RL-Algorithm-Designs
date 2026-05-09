import gymnasium as gym
import torch
from network import A2C_Actor

ENV_NAME = "HalfCheetah-v5"
MODEL_PATH = "models/ppo_actor_best.pth"
DEVICE = "cpu" # Test işlemi CPU'da yeterince hızlı

print("Şampiyon PPO Ajanı Sahaya Çıkıyor...")


env = gym.make(ENV_NAME, render_mode="human")
obs_shape = env.observation_space.shape
n_actions = env.action_space.shape[0]

# Actor Ağını Yükle
actor = A2C_Actor(obs_shape, n_actions).to(DEVICE)
actor.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
actor.eval() # Ağı "Değerlendirme/Test" moduna al (BatchNorm/Dropout varsa etkilenmesin diye)

for episode in range(5):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Durumu Tensöre çevir
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        # Gradyan hesaplamalarını kapat (Çok daha hızlı çalışır)
        with torch.no_grad():
            # Ağı çalıştır. Sadece merkezi (mu) al, standart sapmayı (std) umursama
            mu, _ = actor(state_t)
            
            # Limitlere göre kırp ve NumPy'a çevir
            action = torch.clamp(mu, -1.0, 1.0).cpu().numpy()[0]
            
        # Hareketi simülasyona gönder
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
    print(f"Bölüm {episode + 1} Tamamlandı | Toplam Skor: {episode_reward:.1f}")

env.close()
print("Test Bitti.")