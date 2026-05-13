import gymnasium as gym
import numpy as np
import torch
import os
import random
from torch.utils.tensorboard import SummaryWriter


from network import SAC_Actor, SAC_Critic
from agent import SACAgent

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Yeni bir anıyı hafızaya ekle"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Hafızadan rastgele bir 'anı yığını' (batch) çek"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

ENV_NAME = "HalfCheetah-v5"
MAX_EPISODES = 2000
BATCH_SIZE = 256
START_STEPS = 10000 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"SAC (Maksimum Entropi) Başlatıldı! Cihaz: {DEVICE}")

env = gym.make(ENV_NAME)
obs_shape = env.observation_space.shape
n_actions = env.action_space.shape[0]


actor = SAC_Actor(obs_shape, n_actions)
critic = SAC_Critic(obs_shape, n_actions)
agent = SACAgent(actor, critic, device=DEVICE, action_dim=n_actions)
replay_buffer = ReplayBuffer(capacity=1000000)

writer = SummaryWriter(log_dir="runs/SAC_Continuous")

best_reward = -np.inf
rewards_history = []
total_steps = 0

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # HAREKET SEÇİMİ (Başlangıçta rastgele, sonra SAC'ın kaotik beyni)
        if total_steps < START_STEPS:
            action = env.action_space.sample() # Havuzu doldurmak için tamamen rastgele
        else:
            action = agent.select_action(state, evaluate=False) # Reparameterization devrede
            
        # SİMÜLASYONA GÖNDER
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        #  ANIYI HAFIZAYA YAZ
        # PPO'daki gibi listelerde tutup bölüm sonu silmiyoruz. Her şey kalıcı havuza gidiyor.
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        total_steps += 1
        
        # Her adımda hafızadan 256 anı çek ve ağı güncelle
        if len(replay_buffer) > BATCH_SIZE and total_steps > START_STEPS:
            batch = replay_buffer.sample(BATCH_SIZE)
            critic_loss, actor_loss, alpha = agent.update(batch)
            
            writer.add_scalar("Loss/Critic", critic_loss, total_steps)
            writer.add_scalar("Loss/Actor", actor_loss, total_steps)
            writer.add_scalar("Metrikler/Alpha_Sicakligi", alpha, total_steps)


    rewards_history.append(episode_reward)
    recent_avg = np.mean(rewards_history[-10:])
    writer.add_scalar("Reward/Episode", episode_reward, episode)
    writer.add_scalar("Reward/Average_10", recent_avg, episode)
    

    if recent_avg > best_reward and total_steps > START_STEPS:
        best_reward = recent_avg
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(agent.actor.state_dict(), "models/sac_actor_best.pth")
        print(f"Yeni Rekor! Model Kaydedildi: {best_reward:.1f}")
    
    # İlk 10.000 adımda log basmayı sade tutalım (Henüz ajan öğrenmiyor)
    if total_steps < START_STEPS:
        print(f"Bölüm: {episode+1:4d} | Havuz Doluyor... Adım: {total_steps}/{START_STEPS}")
    else:
        print(f"Bölüm: {episode+1:4d} | Skor: {episode_reward:7.1f} | Ort: {recent_avg:7.1f} | Adım: {total_steps} | Alpha: {agent.alpha:7.4f}")

env.close()
writer.close()