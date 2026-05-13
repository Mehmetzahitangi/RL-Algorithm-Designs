import gymnasium as gym
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter


from count_wrapper import PseudoCountRewardWrapper
#from network import PPONetwork
from agent import PPOAgent

from noisy_ppo_network import  NoisyPPONetwork # Noisy Network için diğeri kapatır bunu açarız


ENV_NAME = "MountainCar-v0"
PPO_STEPS = 2048 # PPO'nun bir seferde oynayacağı maksimum adım
MAX_EPISODES = 50000
GAMMA = 0.99
LAMBDA = 0.95 # GAE (Generalized Advantage Estimation) için
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f" Modüler PPO + Count-Based Motoru Başlatıldı! Cihaz: {DEVICE}")

base_env = gym.make(ENV_NAME)
env = PseudoCountRewardWrapper(base_env, reward_scale=1.0)

obs_shape = env.observation_space.shape
n_actions = env.action_space.n

network = NoisyPPONetwork(obs_shape, n_actions)
agent = PPOAgent(network, lr=3e-4, device=DEVICE)

writer = SummaryWriter(log_dir="runs/NoisyPPO_MountainCar") # CountBasedPPO_MountainCar

if not os.path.exists("models"):
    os.makedirs("models")

best_real_reward = -np.inf
rewards_history = []
real_rewards_history = []
global_step = 0
episode_count = 0

state, _ = env.reset()
ep_reward_curiosity = 0
ep_reward_real = 0

while episode_count < MAX_EPISODES:
    #  ROLLOUT (VERİ) TOPLAMA AŞAMASI 
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

    network.sample_noise() # noisy nework için bunu açarız
    
    for _ in range(PPO_STEPS):
        action, log_prob, value = agent.select_action(state)
        next_state, total_r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        

        states.append(state)
        actions.append(action)
        rewards.append(total_r) # Merak (Intrinsic) eklenmiş ödül
        values.append(value)
        log_probs.append(log_prob)
        dones.append(done)
        
        state = next_state
        ep_reward_curiosity += total_r
        ep_reward_real += -1.0 # Gerçek oyun skoru
        global_step += 1
        
        if done:
            episode_count += 1
            rewards_history.append(ep_reward_curiosity)
            real_rewards_history.append(ep_reward_real)
            
            recent_avg_real = np.mean(real_rewards_history[-100:])
            
            writer.add_scalar("Reward/Total_with_Curiosity", ep_reward_curiosity, episode_count)
            writer.add_scalar("Reward/Real_Game_Score", ep_reward_real, episode_count)
            writer.add_scalar("Reward/Average_100_Real", recent_avg_real, episode_count)
            
            # -200'ü geçmek hedefi bulmaktır
            if ep_reward_real > -200 and recent_avg_real > best_real_reward:
                best_real_reward = recent_avg_real
                torch.save(network.state_dict(), "models/noisy_ppo_best.pth") # count_ppo_best
                print(f"Yeni Rekor. Model Kaydedildi: {best_real_reward:.1f} (Bölüm: {episode_count})")
            
            if episode_count % 50 == 0:
                print(f"Bölüm: {episode_count:5d} | Gerçek Skor: {ep_reward_real:7.1f} | Meraklı Skor: {ep_reward_curiosity:7.1f}")
                
            state, _ = env.reset()
            ep_reward_curiosity = 0
            ep_reward_real = 0

    #  GAE - Avantaj Hesaplama 
    _, _, next_value = agent.select_action(state)
    returns = []
    advantages = []
    gae = 0
    
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_v = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_v = values[step + 1]
            
        delta = rewards[step] + GAMMA * next_v * next_non_terminal - values[step]
        gae = delta + GAMMA * LAMBDA * next_non_terminal * gae
        
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
        
    rollouts = (states, actions, log_probs, returns, advantages)
    
    actor_loss, critic_loss, entropy = agent.update(rollouts)
    
    writer.add_scalar("Loss/Actor", actor_loss, global_step)
    writer.add_scalar("Loss/Critic", critic_loss, global_step)
    writer.add_scalar("Metrikler/Entropy", entropy, global_step)

env.close()
writer.close()