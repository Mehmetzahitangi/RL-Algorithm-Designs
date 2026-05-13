import gymnasium as gym
import torch
from network import SAC_Actor

ENV_NAME = "HalfCheetah-v5"
MODEL_PATH = "models/sac_actor_best.pth"
DEVICE = "cpu"

print("SAC (Maksimum Entropi) Başlatılıyor ...")


env = gym.make(ENV_NAME, render_mode="human")
obs_shape = env.observation_space.shape
n_actions = env.action_space.shape[0]


actor = SAC_Actor(obs_shape, n_actions).to(DEVICE)
actor.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
actor.eval() 

for episode in range(5):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # SAC'ın sihirli noktası: 
            # 1. ve 2. çıktıyı (gürültülü aksiyon ve ihtimal) yoksay.
            # 3. çıktıyı (saf, net aksiyon olan mean_action) al!
            _, _, mean_action = actor.sample(state_t)
            
            # Aksiyonu NumPy dizisine çevir (Simülasyon motoru için)
            action = mean_action.cpu().numpy()[0]
            
        # Hareketi simülasyona gönder
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
    print(f"Bölüm {episode + 1} Tamamlandı | Toplam Skor: {episode_reward:.1f}")

env.close()
print("Test Bitti.")