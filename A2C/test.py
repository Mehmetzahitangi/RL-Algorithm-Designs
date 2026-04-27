import torch
import time
from agent import ReinforceAgent
import gymnasium as gym

def test_agent():

    print("Ortam Yükleniyor...")
    
    env = gym.make('LunarLander-v2', render_mode='human')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ReinforceAgent(obs_size, n_actions)

    agent.policy_net.load_state_dict(torch.load("a2c_lunarlander_best.pth"))
    #agent.policy_net.load_state_dict(torch.load("A2C_cartpole_best.pth"))
    agent.policy_net.eval()

    for episode in range(5): # 5 oyun boyunca izlemek için
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action ,_, _, _= agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(0.02)
            
        print(f"Test Bölümü {episode + 1} bitti. Toplam Ödül: {total_reward}")

    env.close()

if __name__ == "__main__":
    test_agent()