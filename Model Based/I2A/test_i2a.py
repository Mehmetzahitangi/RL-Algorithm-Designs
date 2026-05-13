import torch
import torch.nn.functional as F
import gymnasium as gym
import time


from env_model import EnvironmentModel
from i2a_components import RolloutPolicy, RolloutEncoder
from i2a_model import I2A_Agent


ENV_NAME = "BreakoutNoFrameskip-v4"
OBS_SHAPE = (4, 84, 84) # Frame stack
N_ACTIONS = 4
MODEL_PATH = "models/i2a_best_breakout.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Test Ortamı Başlıyor... Hedef: {MODEL_PATH}")


env = gym.make(ENV_NAME, render_mode="human") # Atari Wrapper


# Mimarinin İçini Boş Olarak İnşa Et
env_model = EnvironmentModel(OBS_SHAPE, N_ACTIONS).to(DEVICE)
rollout_policy = RolloutPolicy(OBS_SHAPE, N_ACTIONS).to(DEVICE)
rollout_encoder = RolloutEncoder(OBS_SHAPE, hidden_size=256).to(DEVICE)

# Ana Beyni Oluştur ve Ağırlıkları Yükle
i2a_agent = I2A_Agent(OBS_SHAPE, N_ACTIONS, env_model, rollout_policy, rollout_encoder).to(DEVICE)

# Eğitilmiş beyin hücresi ağırlıklarını modele enjekte et
try:
    i2a_agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Ağırlıklar başarıyla yüklendi")
except Exception as e:
    print(f"Model yüklenemedi. Hata: {e}")
    exit()

# Eğitim modundan Çıkarım (Inference) moduna al (Çok önemlidir, Dropout/BatchNorm gibi katmanları sabitler)
i2a_agent.eval()

state, _ = env.reset()
done = False
total_score = 0

while not done:
    # Ajanı ekranda izleyebilmek için hafif bir gecikme ekleyebiliriz
    time.sleep(0.02)
    
    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Ajan kendi içinde hayaller kurup bize en mantıklı kararı döner
        logits, _ = i2a_agent(state_t)
        
        # Test sırasında rastgeleliği (exploration) kapatırız.
        # En yüksek ihtimalli hamleyi (Greedy) seçeriz!
        action = logits.argmax(dim=1).item()
        
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_score += reward

print(f"Oyun Bitti! Toplam Skor: {total_score}")
env.close()