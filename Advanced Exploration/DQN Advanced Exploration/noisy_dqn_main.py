import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from noisy_dqn_network import NoisyDQN
from noisy_dqn_agent import NoisyDQNAgent, ReplayBuffer
import os 

ENV_NAME = "MountainCar-v0"
MAX_EPISODES = 30000
BATCH_SIZE = 128
SYNC_TARGET_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Noisy DQN Başlatıldı! Cihaz: {DEVICE}")

env = gym.make(ENV_NAME)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n


# Ağ, Ajan ve Havuz
net = NoisyDQN(obs_shape, n_actions)
agent = NoisyDQNAgent(net, lr=1e-3, device=DEVICE)
buffer = ReplayBuffer(capacity=100000)

writer = SummaryWriter(log_dir="runs/NoisyDQN_MountainCar")

total_steps = 0
best_reward = -np.inf
rewards_history = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        total_steps += 1
        
        if len(buffer) > BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            loss = agent.update(batch)
            
            if total_steps % SYNC_TARGET_STEPS == 0:
                agent.sync_target()
                
            writer.add_scalar("Loss/Q_Value", loss, total_steps)

    rewards_history.append(episode_reward)
    recent_avg = np.mean(rewards_history[-100:])
    
    writer.add_scalar("Reward/Episode", episode_reward, episode)
    writer.add_scalar("Reward/Average_100", recent_avg, episode)
    

    if recent_avg > best_reward and len(rewards_history) >= 100:
        best_reward = recent_avg
        if not os.path.exists("models"):
            os.makedirs("models")
        # Gürültülü ağırlıkları kaydet!
        torch.save(net.state_dict(), "models/noisy_dqn_mcar_best.pth")
        
    if episode_reward > -200:
        print(f"HEDEF BULUNDU! Bölüm: {episode+1} | Skor: {episode_reward:.1f}")
        
    if (episode + 1) % 100 == 0:
        print(f"Bölüm: {episode+1:5d} | Son 100 Ortalaması: {recent_avg:7.1f} | En İyi Ort: {best_reward:7.1f}")

env.close()
writer.close()