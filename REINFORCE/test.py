import torch
import time
from agent import ReinforceAgent
import gymnasium as gym

def test_agent():
    print("Ortam Yükleniyor...")
    # render_mode='human' diyerek simülasyon penceresini açıyoruz
    #env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('LunarLander-v2', render_mode='human')

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # Ajanı sanki eğitime baştan başlayacakmış gibi boş oluşturuyoruz
    agent = ReinforceAgent(obs_size, n_actions)

    agent.policy_net.load_state_dict(torch.load("reinforce_lunarlander_entropy_best.pth"))
    #agent.policy_net.load_state_dict(torch.load("reinforce_cartpole_best.pth"))
    agent.policy_net.eval()

    for episode in range(5): # 5 oyun boyunca izlemek için
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            #entropy kullanılan eğitimler için
            action, _ , _= agent.select_action(state)
            
            #action, _ = agent.select_action(state)
            
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Rahat görebilmemiz için biraz yavaşlatıyoruz
            time.sleep(0.02) 
            
        print(f"Test Bölümü {episode + 1} bitti. Toplam Ödül: {total_reward}")

    env.close()

if __name__ == "__main__":
    test_agent()