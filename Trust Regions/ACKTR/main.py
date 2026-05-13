import gymnasium as gym
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from network import ACKTR_Actor, ACKTR_Critic
from acktr_agent import ACKTRAgent


ENV_NAME = "HalfCheetah-v5"
MAX_EPISODES = 2000
GAMMA = 0.99
LAMBDA = 0.95 # GAE Yumuşatma faktörü
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"ACKTR (K-FAC) Motoru Başlatıldı! Cihaz: {DEVICE}")

env = gym.make(ENV_NAME)
obs_shape = env.observation_space.shape
n_actions = env.action_space.shape[0]


actor = ACKTR_Actor(obs_shape, n_actions)
critic = ACKTR_Critic(obs_shape)
# ACKTR'nin Öğrenme Oranı (actor_lr) K-FAC'ın içinde 0.25 olarak gizlidir
agent = ACKTRAgent(actor, critic, critic_lr=1e-3, device=DEVICE)

writer = SummaryWriter(log_dir="runs/ACKTR_Continuous")

best_reward = -np.inf
rewards_history = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    
    # Hafıza (Rollout Buffer)
    states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []
    
    episode_reward = 0
    done = False
    
    while not done:
        action_env, action_raw, log_prob, entropy, value = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action_env)
        done = terminated or truncated
        
        states.append(state)
        actions.append(action_raw) # Ağa ham (kırpılmamış) aksiyonu veriyoruz
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)
        
        state = next_state
        episode_reward += reward

    # GAE (Generalized Advantage Estimation HESAPLAMASI
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_val = 0
        else:
            next_val = values[i + 1]
            
        delta = rewards[i] + GAMMA * next_val - values[i]
        gae = delta + GAMMA * LAMBDA * gae
        advantages.insert(0, gae)
        
    returns = [adv + val for adv, val in zip(advantages, values)]

    states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
    actions_t = torch.FloatTensor(np.array(actions)).to(DEVICE)
    returns_t = torch.FloatTensor(returns).to(DEVICE)
    
    log_probs_t = torch.cat(log_probs)
    values_t = torch.cat(values)
    entropies_t = torch.cat(entropies)
    advantages_t = torch.FloatTensor(advantages).to(DEVICE)
    
    # Avantajları Normalize Et (Stabilite için kritik)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    rollouts = {
        'states': states_t,
        'actions': actions_t,
        'log_probs': log_probs_t,
        'returns': returns_t,
        'values': values_t,
        'advantages': advantages_t,
        'entropies': entropies_t
    }
    
    #  AJANI GÜNCELLE (ACKTR Tek Hamlede Matris Tersi Alır)
    critic_loss, actor_loss = agent.update(rollouts)
    
    rewards_history.append(episode_reward)
    recent_avg = np.mean(rewards_history[-10:])
    
    writer.add_scalar("Loss/Critic", critic_loss, episode)
    writer.add_scalar("Loss/Actor", actor_loss, episode)
    writer.add_scalar("Reward/Episode", episode_reward, episode)
    writer.add_scalar("Reward/Average_10", recent_avg, episode)
    
    if recent_avg > best_reward:
        best_reward = recent_avg
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(agent.actor.state_dict(), "models/acktr_actor_best.pth")
        print(f"Yeni Rekor! Model Kaydedildi: {best_reward:.1f}")
    
    print(f"Bölüm: {episode+1:4d} | Skor: {episode_reward:7.1f} | Ort: {recent_avg:7.1f} | A_Loss: {actor_loss:7.4f} | C_Loss: {critic_loss:7.2f}")

env.close()
writer.close()