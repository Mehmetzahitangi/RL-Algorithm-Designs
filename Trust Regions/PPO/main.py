import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from network import A2C_Actor, A2C_Critic
from agent import PPOAgent
import os

ENV_NAME = "HalfCheetah-v5"
MAX_EPISODES = 2000
GAMMA = 0.99
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    env = gym.make(ENV_NAME)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.shape[0]

    # Ağları ve Ajanı Başlat
    actor = A2C_Actor(obs_shape, n_actions)
    critic = A2C_Critic(obs_shape)
    agent = PPOAgent(actor, critic, actor_lr=3e-4, critic_lr=1e-3, device=DEVICE)

    writer = SummaryWriter(log_dir="runs/PPO_Continuous_Improved")

    print("Eğitim Başlıyor")

    best_reward = -np.inf
    rewards_history = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()

        # Bu bölüm için hafıza (Rollout Buffer)
        states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []
    
        episode_reward = 0
        done = False

        # 1. Veri Toplama
        while not done:
                action_env, action_raw, log_prob, entropy, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action_env)
                done = terminated or truncated
                
                # HAFIZAYA HAM (action_raw) OLANINI KAYDET (PPO'nun düzgün çalışabilmesi için kritik)
                states.append(state)
                actions.append(action_raw)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
                
                state = next_state
                episode_reward += reward
                

        # GAE (Generalized Advantage Estimation) HESAPLAMASI
        advantages = []
        gae = 0
        lam = 0.95 # GAE'nin yumuşatma parametresi


        # Sondan başa doğru TD-Error (Delta) hesapla
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = 0 # Bölüm bittiği için sonraki değer 0
            else:
                next_val = values[i + 1]
                
            # TD Hatası (Gerçekleşen Ödül + Gelecek Tahmini - Şimdiki Tahmin)
            delta = rewards[i] + GAMMA * next_val - values[i]
            
            # GAE Formülü: Geçmişteki avantajları lambda ile zayıflatarak bugüne ekle
            gae = delta + GAMMA * lam * gae
            advantages.insert(0, gae)

        # Getiri (Return) = Avantaj + Critic'in Tahmini (V(s))
        returns = [adv + val for adv, val in zip(advantages, values)]
            
        # Listeleri PyTorch Tensörüne Çevir
        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions_t = torch.FloatTensor(np.array(actions)).to(DEVICE)
        returns_t = torch.FloatTensor(returns).to(DEVICE)

        # Ağdan (GPU'dan) doğrudan gelen Tensörleri birleştir (Bunlar zaten "DEVICE" üzerinde)
        log_probs_t = torch.cat(log_probs)
        values_t = torch.cat(values)
        entropies_t = torch.cat(entropies)
        advantages_t = torch.FloatTensor(advantages).to(DEVICE)
        
        # Avantaj = Gerçekleşen Getiri - Critic'in Tahmini
        #advantages_t = returns_t - values_t.detach() # Detach yapıyoruz ki avantaj üzerinden critic'e türev akmasın
        
        # Avantajları Normalize Et (Eğitimi çok hızlandırır ve stabilize eder)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Ajanın anlayacağı paketi hazırla
        rollouts = {
            'states': states_t,
            'actions': actions_t,
            'log_probs': log_probs_t,
            'returns': returns_t,
            'values': values_t,
            'advantages': advantages_t,
            'entropies': entropies_t
        }
        
        # 3. AJANI GÜNCELLE
        critic_loss, actor_loss = agent.update(rollouts, batch_size=64)
        
        # --- LOGLAMA ---
        rewards_history.append(episode_reward)
        recent_avg = np.mean(rewards_history[-10:])
        
        writer.add_scalar("Loss/Critic", critic_loss, episode)
        writer.add_scalar("Loss/Actor", actor_loss, episode)
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Reward/Average_10", recent_avg, episode)
        
        print(f"Bölüm: {episode+1:4d} | Skor: {episode_reward:7.1f} | Ort: {recent_avg:7.1f} | A_Loss: {actor_loss:7.2f} | C_Loss: {critic_loss:7.2f}")

        # --- MODEL KAYDETME (ŞAMPİYON TAKİBİ) ---
        # Son 10 bölümün ortalaması rekor kırdıysa kaydet
        if recent_avg > best_reward:
            best_reward = recent_avg
            # Modelleri kaydetmek için klasör oluştur (yoksa)
            if not os.path.exists("models"):
                os.makedirs("models")
                
            torch.save(agent.actor.state_dict(), "models/ppo_actor_best.pth")
            torch.save(agent.critic.state_dict(), "models/ppo_critic_best.pth")
            print(f"Yeni Rekor! Model Kaydedildi: {best_reward:.1f}")

    env.close()
    writer.close()
    print("Eğitim Tamamlandı")



if __name__ == "__main__":
    train()