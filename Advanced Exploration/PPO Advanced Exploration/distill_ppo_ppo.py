import gymnasium as gym
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter

# Daha önce yazdığımız standart PPO Ağı ve Ajanı (Gürültüsüz, saf PPO)
from network import PPONetwork
from agent import PPOAgent
# Yeni Damıtma (Merak) Motorumuz
from net_distillery import MountainCarNetDistillery


ENV_NAME = "MountainCar-v0"
PPO_STEPS = 2048
MAX_EPISODES = 70000
GAMMA = 0.99
LAMBDA = 0.95
DEVICE = "cpu" # Distillation basit ağlar olduğu için CPU'da da çok hızlı çalışır

print("Network Distillation (RND) Başlatıldı!")

env = gym.make(ENV_NAME)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n


network = PPONetwork(obs_shape, n_actions)
agent = PPOAgent(network, lr=3e-4, device=DEVICE)

# Distillation Başlat
distillery = MountainCarNetDistillery(obs_shape[0]).to(DEVICE)
distill_optimizer = torch.optim.Adam(distillery.trn_net.parameters(), lr=1e-4)

writer = SummaryWriter(log_dir="runs/DistillPPO_MountainCar")

if not os.path.exists("models"):
    os.makedirs("models")

best_real_reward = -np.inf
global_step = 0
episode_count = 0
rewards_history, real_rewards_history = [], []

state, _ = env.reset()
ep_reward_curiosity = 0
ep_reward_real = 0

while episode_count < MAX_EPISODES:
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    
    #  ROLLOUT (Deneyim Toplama ve Merak Puanı Üretme) 
    for _ in range(PPO_STEPS):
        action, log_prob, value = agent.select_action(state)
        next_state, real_r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Distillation: Merak Puanını (Tahmin Hatasını) Al
        intrinsic_reward = distillery.extra_reward(state)
        
        # PPO'nun beynine gidecek olan toplam ödül (Oyun skoru + Merak)
        total_r = real_r + intrinsic_reward 
        
        states.append(state)
        actions.append(action)
        rewards.append(total_r)
        values.append(value)
        log_probs.append(log_prob)
        dones.append(done)
        
        state = next_state
        ep_reward_curiosity += total_r
        ep_reward_real += real_r
        global_step += 1
        
        if done:
            episode_count += 1
            real_rewards_history.append(ep_reward_real)
            recent_avg = np.mean(real_rewards_history[-100:])
            
            writer.add_scalar("Reward/Total_with_Curiosity", ep_reward_curiosity, episode_count)
            writer.add_scalar("Reward/Real_Game_Score", ep_reward_real, episode_count)
            writer.add_scalar("Reward/Average_100", recent_avg, episode_count)
            
            if ep_reward_real > -200 and recent_avg > best_real_reward:
                best_real_reward = recent_avg
                torch.save(network.state_dict(), "models/distill_ppo_best.pth")
                print(f"Yeni Rekor! Model Kaydedildi: {best_real_reward:.1f}")
                
            if episode_count % 50 == 0:
                print(f"Bölüm: {episode_count:5d} | Gerçek Skor: {ep_reward_real:7.1f} | Meraklı Skor: {ep_reward_curiosity:7.1f}")
                
            state, _ = env.reset()
            ep_reward_curiosity = 0
            ep_reward_real = 0

    #  (GAE) 
    _, _, next_value = agent.select_action(state)
    returns, advantages, gae = [], [], 0
    for step in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[-1] if step == len(rewards) - 1 else 1.0 - dones[step]
        next_v = next_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + GAMMA * next_v * next_non_terminal - values[step]
        gae = delta + GAMMA * LAMBDA * next_non_terminal * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
        
    rollouts = (states, actions, log_probs, returns, advantages)
    
    actor_loss, critic_loss, entropy = agent.update(rollouts)
    
    # Distilattion Güncelleme  (Taklitçi ağın kendini geliştirmesi için)
    states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
    distill_loss = distillery.loss(states_t)
    
    distill_optimizer.zero_grad()
    distill_loss.backward()
    distill_optimizer.step()
    
    # Loglama
    writer.add_scalar("Loss/Actor", actor_loss, global_step)
    writer.add_scalar("Loss/Distillation (Merak Algisi)", distill_loss.item(), global_step)

env.close()
writer.close()