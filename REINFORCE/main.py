# Simülasyon ortamı (Environment) ve eğitim döngüsü

import gymnasium as gym
from agent import ReinforceAgent
import torch
from utils import plot_learning_curve

MAX_EPISODES = 4000 #1000

def train():
    # 1. Ortamı Kur

    #env = gym.make('CartPole-v1')
    #obs_size = env.observation_space.shape[0]  # Çıktı: 4 boyutlu durum
    #n_actions = env.action_space.n             # Çıktı: 2 aksiyon (Sağ/Sol)

    env = gym.make('LunarLander-v2')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n            
    # LunarLander için tavsiyeler: Episode sayısını 3-5 bin arası yapmak, ağın nöron sayısını artırmak, bölümler daha uzun sürdüğü için episode %8 == 0 azaltmak 

    # 2. Ajanı Başlat
    agent = ReinforceAgent(obs_size, n_actions)

    print("Eğitim Başlıyor")

    best_reward = 0.0
    batch_log_probs = []
    batch_returns = []
    all_episode_rewards = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # 1. Veri Toplama
        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, done ,truncated ,_ = env.step(action)
            
            
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done or truncated:
                break
                

        returns = agent.calculate_returns(rewards)
        batch_returns.extend(returns)
        batch_log_probs.extend(log_probs)

        if episode %4 == 0 and episode > 0:
            agent.update_policy(batch_log_probs, batch_returns)
            batch_log_probs = []
            batch_returns = []

        total_reward = sum(rewards)
        all_episode_rewards.append(total_reward)

        if total_reward >= best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), "reinforce_lunarlander_best.pth")
            print(f"Yeni Rekor! Bölüm: {episode} | Puan: {best_reward} -> Model kaydedildi.")

        if episode % 20 == 0:
            print(f"Bölüm (Episode): {episode:3d} | Toplam Ödül: {total_reward:5.1f}")


    env.close()
    print("Eğitim Tamamlandı")

    plot_learning_curve(all_episode_rewards)


if __name__ == "__main__":
    train()