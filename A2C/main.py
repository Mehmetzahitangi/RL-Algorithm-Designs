import gymnasium as gym
from agent import ReinforceAgent
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from atari_wrappers import make_env

def train():

    #env = gym.make('CartPole-v1')
    #obs_size = env.observation_space.shape[0]  # Çıktı: 4 boyutlu durum
    #n_actions = env.action_space.n             # Çıktı: 2 aksiyon (Sağ/Sol)

    NUM_ENVS = 16
    N_STEPS = 128 # 128 adımlık Rollout (N-Step)
    MAX_UPDATES = 1000

    envs = gym.make_vec('LunarLander-v2', num_envs=NUM_ENVS) # (16, 8) 
    obs_size = envs.single_observation_space.shape[0] 
    n_actions = envs.single_action_space.n  

    #envs = gym.vector.SyncVectorEnv([lambda: make_env("ALE/Pong-v5") for _ in range(NUM_ENVS)]) # Görsel atari ortamı için
    #obs_size = envs.single_observation_space.shape  # Görsel ortam için
    #n_actions = envs.single_action_space.n

    print(f"Tekil Gözlem Boyutu: {obs_size}")
    print(f"Tekil Eylem Sayısı: {n_actions}")

    agent = ReinforceAgent(obs_size, n_actions, lr=7e-4) # atari ortamı için 1e-4 veya 2.5e-4. Lunarlander ve diğerleri için 7e-4,
    state, _ = envs.reset()

    print("Eğitim Başlıyor")

    writer = SummaryWriter("logs/deneme_a2c")

    all_episode_rewards = []
    best_reward = -1000.0 # LunarLander eksi puanlarla başladığı için -1000 yapılabilir


    # Başlangıç değeri
    entropy_beta = 0.01 
    # Her güncellemede ne kadar azalacağı
    entropy_decay = 0.995 
    # Minimum düşeceği seviye (Tamamen sıfırlanmasın biraz merak ögesi kalsın)
    min_beta = 0.001

    current_episode_rewards = np.zeros(NUM_ENVS)
    true_scores = []

    for update in range(MAX_UPDATES):

        log_probs, values, rewards, entropies, masks = [], [], [], [], []
        
        for step in range(N_STEPS):
            action, log_prob, entropy, value = agent.select_action(state)
            next_state, reward, done ,truncated ,_ = envs.step(action)
            
            # MASKELEME (Ölüm Kontrolü)
            mask = 1.0 - done

            current_episode_rewards += reward

            for i in range(NUM_ENVS):
                if done[i] or truncated[i]:
                    # Final skorunu arşive kaydet
                    true_scores.append(current_episode_rewards[i])
                    # Yeniden başladığı için o ajanın sayacını sıfırla
                    current_episode_rewards[i] = 0.0

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            masks.append(mask)

            state = next_state
                

        # Bootstrapping 128 adım tamamlandıktan sonra son durumu vererek geleceği tahmin ediyoruz
        next_state_tensor = torch.from_numpy(next_state).float().to(agent.device)
        _, next_value = agent.policy_net(next_state_tensor)

        returns = agent.calculate_returns(rewards, masks, next_value.detach())
        total_loss, a_loss, c_loss, ent_loss = agent.update_policy(log_probs, values, returns, entropies)

        writer.add_scalar("Kayıp_Detay/1_Toplam", total_loss, update)
        writer.add_scalar("Kayıp_Detay/2_Actor", a_loss, update)
        writer.add_scalar("Kayıp_Detay/3_Critic", c_loss, update)
        writer.add_scalar("Kayıp_Detay/4_Entropy", ent_loss, update)
        writer.add_scalar("Parametre/Entropy_Beta", entropy_beta, update)

        # REKOR KONTROLÜNÜ ARTIK "GERÇEK" BİTEN OYUNLAR ÜZERİNDEN YAPIYORUZ
        # Eğer en az 1 oyun bittiyse ortalamasına bak (Son 20 oyunun ortalaması daha sağlıklıdır)
        if len(true_scores) > 0:
            mean_score = np.mean(true_scores[-20:])
            
            if mean_score > best_reward:
                best_reward = mean_score
                torch.save(agent.policy_net.state_dict(), "a2c_deneme_best.pth")
                print(f"Yeni Rekor! Update: {update:3d} | Gerçek Ortalama Puan: {best_reward:.1f} -> Model kaydedildi.")
                
            # TensorBoard'a artık bu gerçek skoru gönderiyoruz
            writer.add_scalar("Ödül/Gerçek_Ortalama", mean_score, update)

        entropy_beta = max(min_beta, entropy_beta * entropy_decay) # Beta güncelleme, DECAY Rate
        #if episode %4 == 0 and episode > 0: ve yukarıdaki verileri batch olarak tutmaya artık gerek yok 16 ajandan 128 adımlık yeterli sayıda veri geliyor
        
    envs.close()
    print("Eğitim Tamamlandı")


if __name__ == "__main__":
    train()