import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.tensorboard import SummaryWriter
from count_wrapper import PseudoCountRewardWrapper
import os

#  BASİT DQN AĞI (Noisy Yok) 
class SimpleDQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(SimpleDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

#  STANDART REPLAY BUFFER 
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


ENV_NAME = "MountainCar-v0"
MAX_EPISODES = 15000
BATCH_SIZE = 128
GAMMA = 0.99
SYNC_TARGET_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Count-Based Başlatıldı! Cihaz: {DEVICE}")

# Ortamı Oluştur Wrapper'ı Uygula
base_env = gym.make(ENV_NAME)
env = PseudoCountRewardWrapper(base_env, reward_scale=1.0)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n

net = SimpleDQN(obs_shape, n_actions).to(DEVICE)
target_net = SimpleDQN(obs_shape, n_actions).to(DEVICE)
target_net.load_state_dict(net.state_dict())
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
buffer = ReplayBuffer(100000)

writer = SummaryWriter(log_dir="runs/CountBasedDQN_MountainCar")

if not os.path.exists("models"):
    os.makedirs("models")
best_real_reward = -np.inf

total_steps = 0
epsilon = 1.0
epsilon_decay = 0.99995 # Epsilon yavaşça azalacak
epsilon_min = 0.05
rewards_history = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0 # (Extrinsic + Intrinsic)
    real_reward = 0    # Sadece oyunun gerçek skoru
    done = False
    
    while not done:
        # Epsilon-Greedy Seçim, agent.select_action(state) nin yerine geçen kısım
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = net(state_t).max(1)[1].item()
                
        # Simülasyon Adımı (Wrapper bize "Merak" eklenmiş ödülü dönecek)
        next_state, total_r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.push(state, action, total_r, next_state, done)
        state = next_state
        episode_reward += total_r
        real_reward += -1.0 # MountainCar'da gerçek ödül her zaman -1'dir
        total_steps += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Eğitim, agent.update(batch)'in yerini tutacak kısım
        if len(buffer) > BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            s, a, r, next_s, d = batch
            
            s_t = torch.FloatTensor(s).to(DEVICE)
            a_t = torch.LongTensor(a).to(DEVICE)
            r_t = torch.FloatTensor(r).to(DEVICE)
            next_s_t = torch.FloatTensor(next_s).to(DEVICE)
            d_t = torch.FloatTensor(d).to(DEVICE)
            
            q_values = net(s_t).gather(1, a_t.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_q_values = target_net(next_s_t).max(1)[0]
                expected_q = r_t + GAMMA * next_q_values * (1 - d_t)
                
            loss = nn.MSELoss()(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if total_steps % SYNC_TARGET_STEPS == 0:
                target_net.load_state_dict(net.state_dict())


    rewards_history.append(real_reward)
    recent_avg = np.mean(rewards_history[-100:])
    
    writer.add_scalar("Reward/Total_with_Curiosity", episode_reward, episode)
    writer.add_scalar("Reward/Real_Game_Score", real_reward, episode)
    writer.add_scalar("Reward/Average_100_Real", recent_avg, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    # MountainCar'da -200 barajını kırmak hedefe ulaşmak demektir.
    if real_reward > -200 and recent_avg > best_real_reward:
        best_real_reward = recent_avg
        torch.save(net.state_dict(), "models/count_dqn_best.pth")
        print(f"Yeni Rekor! Model Kaydedildi: {best_real_reward:.1f} (Bölüm: {episode+1})")
        
    if (episode + 1) % 100 == 0:
        print(f"Bölüm: {episode+1:5d} | Gerçek Skor: {real_reward:7.1f} | Toplam Skor (Meraklı): {episode_reward:7.1f} | Epsilon: {epsilon:.2f}")

env.close()
writer.close()