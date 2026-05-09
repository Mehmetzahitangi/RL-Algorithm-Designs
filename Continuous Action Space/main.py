import gymnasium as gym
import numpy as np
import torch
from agent import DDPGAgent
from buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def train():

    ENV_NAME = "HalfCheetah-v5"
    MAX_EPISODES = 1000      # Toplam kaç bölüm oynanacak
    MAX_STEPS = 1000         # Her bölüm maksimum kaç adım sürecek
    BATCH_SIZE = 64          # Hafızadan tek seferde çekilecek anı sayısı
    BUFFER_CAPACITY = 100000 # Hafıza kapasitesi

    env = gym.make(ENV_NAME)
    obs_shape = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    print(f"DDPG Başlatıldı Gözlem: {obs_shape}, Motor Sayısı: {n_actions}")

    agent = DDPGAgent(obs_shape, n_actions)
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    best_reward = -np.inf 
    rewards_history = []

    print("Eğitim Başlıyor")

    writer = SummaryWriter(log_dir="runs/DDPG_HalfCheetah_Deney2_D4PG")

    global_step = 0

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):

            global_step += 1
            action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Deneyimi kaydet
            replay_buffer.push(state, action, reward, next_state, done)    

            # Ajanın Ağları güncellenir
            # Ajan her adımda hafızasından 64 deneyim çekip kendi kendini eğitir
            losses = agent.update(replay_buffer, BATCH_SIZE) 

            # Eğer buffer dolduysa ve eğitim başladıysa logla
            if losses[0] is not None:
                #c_loss, a_loss = losses
                writer.add_scalar("Loss/Critic", losses[0], global_step)
                writer.add_scalar("Loss/Actor", losses[1], global_step)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)
        # Son 10 bölümün hareketli ortalamasını alarak gerçek başarıyı ölçüyoruz
        recent_avg_reward = np.mean(rewards_history[-10:])      
    
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Reward/Average_10", recent_avg_reward, episode)
       
        
        print(f"Bölüm: {episode+1:4d} | Skor: {episode_reward:7.1f} | Son 10 Ortalaması: {recent_avg_reward:7.1f}")
        
        if recent_avg_reward > best_reward:
            best_reward = recent_avg_reward
            torch.save(agent.actor.state_dict(), "D4PG_actor_best.pth")
            torch.save(agent.critic.state_dict(), "D4PG_critic_best.pth")
            print(f"Yeni Rekor! Model Kaydedildi: {best_reward:.1f}")

    env.close()
    writer.close()
    print("Eğitim Tamamlandı")


if __name__ == "__main__":
    train()