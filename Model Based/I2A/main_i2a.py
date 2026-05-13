import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from env_model import EnvironmentModel
from i2a_components import RolloutPolicy, RolloutEncoder
from i2a_model import I2A_Agent
from atari_wrappers import make_env



ENV_NAME = "ALE/Breakout-v5" 
env = make_env(ENV_NAME)


MAX_EPISODES = 50000
BATCH_SIZE = 128
OBS_SHAPE = (4, 84, 84) 
N_ACTIONS = 4 
GAMMA = 0.99           # Gelecekteki ödüllerin discountu (A2C için)
ENTROPY_BETA = 0.01    # Keşif (Merak) primi (A2C için)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists("models"):
    os.makedirs("models")

best_reward = -np.inf
rewards_history = []
total_steps = 0

print(f"I2A Başlatıldı. Cihaz: {DEVICE}")

writer = SummaryWriter(log_dir="runs/I2A_Breakout")

env_model = EnvironmentModel(OBS_SHAPE, N_ACTIONS).to(DEVICE)
rollout_policy = RolloutPolicy(OBS_SHAPE, N_ACTIONS).to(DEVICE)
rollout_encoder = RolloutEncoder(OBS_SHAPE, hidden_size=256).to(DEVICE)

i2a_agent = I2A_Agent(OBS_SHAPE, N_ACTIONS, env_model, rollout_policy, rollout_encoder).to(DEVICE)

optimizer_agent = optim.RMSprop(i2a_agent.parameters(), lr=1e-4, eps=1e-5)
optimizer_em = optim.Adam(env_model.parameters(), lr=1e-4)
optimizer_policy = optim.Adam(rollout_policy.parameters(), lr=1e-4)

def train_i2a_step(batch):
    # A2C matematiği için actions_idx ve dones verilerini de batch'e ekledik!
    states, actions_one_hot, actions_idx, rewards, next_states, dones = batch
    
    # HEDEF 1: ÇEVRE MODELİNİ (EM) EĞİT 
    optimizer_em.zero_grad()
    pred_next_states, pred_rewards = env_model(states, actions_one_hot)
    em_obs_loss = F.mse_loss(pred_next_states, next_states)
    em_reward_loss = F.mse_loss(pred_rewards, rewards)
    em_total_loss = em_obs_loss + em_reward_loss
    em_total_loss.backward()
    optimizer_em.step()

    # HEDEF 2: TAKLİTÇİ PİLOTU EĞİT (Damıtma)
    optimizer_policy.zero_grad()
    with torch.no_grad():
        main_logits, _ = i2a_agent(states)
        main_probs = F.softmax(main_logits, dim=1)
        
    rollout_logits = rollout_policy(states)
    rollout_log_probs = F.log_softmax(rollout_logits, dim=1)
    distillation_loss = -(main_probs * rollout_log_probs).sum(dim=1).mean()
    distillation_loss.backward()
    optimizer_policy.step()


    # HEDEF 3: ANA BEYNİN (I2A) EĞİTİMİ (Tamamlanmış A2C)
    optimizer_agent.zero_grad()
    
    # 1. Şu anki durumun (State) Değerleri ve Kararları
    agent_logits, agent_values = i2a_agent(states)
    
    # 2. Gelecekteki durumun (Next State) Değerleri (Bellman denklemi için)
    with torch.no_grad():
        _, next_agent_values = i2a_agent(next_states)
        # Eğer ajan öldüyse (done=1) gelecekteki değer sıfırdır!
        expected_returns = rewards + GAMMA * next_agent_values.squeeze(-1) * (1.0 - dones)
        
    # 3. Avantajın Hesaplanması
    # Beklenilenden ne kadar daha iyi bir sonuç geldi?
    advantages = expected_returns - agent_values.squeeze(-1)
    
    # 4. Aktör (Actor) Kaybı
    log_probs = F.log_softmax(agent_logits, dim=1)
    # Sadece GERÇEKTE yapılan hamlenin (action_idx) log_prob'unu al
    action_log_probs = log_probs.gather(1, actions_idx.unsqueeze(-1)).squeeze(-1)
    # Avantajlı hamlelerin ihtimalini artır (Negatif yapıyoruz ki gradient descent küçültsün)
    actor_loss = -(action_log_probs * advantages.detach()).mean()
    
    # 5. Eleştirmen (Critic) Kaybı
    critic_loss = F.mse_loss(agent_values.squeeze(-1), expected_returns.detach())
    
    # 6. Entropi Bonusu (Ajanın sürekli aynı tuşa basıp ezberlemesini önler)
    probs = F.softmax(agent_logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    
    # 7. TOPLAM I2A KAYBI VE GERİYE YAYILIM
    total_agent_loss = actor_loss + 0.5 * critic_loss - (ENTROPY_BETA * entropy)
    
    total_agent_loss.backward()
    
    # Derin sinir ağlarında gradient explosion önlemek için kesme işlemi
    nn.utils.clip_grad_norm_(i2a_agent.parameters(), 0.5) 
    optimizer_agent.step()
    
    return em_total_loss.item(), distillation_loss.item(), total_agent_loss.item()



for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    b_states, b_actions_one_hot, b_actions_idx, b_rewards, b_next_states, b_dones = [], [], [], [], [], []
    
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits, value = i2a_agent(state_t)
            probs = F.softmax(logits, dim=1)
            action = torch.multinomial(probs, 1).item() 
            
            action_one_hot = torch.zeros(N_ACTIONS)
            action_one_hot[action] = 1.0
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        b_states.append(state)
        b_actions_one_hot.append(action_one_hot.numpy())
        b_actions_idx.append(action) # A2C matematiği için saf indeks
        b_rewards.append(reward) 
        b_next_states.append(next_state)
        b_dones.append(float(done))  # A2C maskesi için 1.0 veya 0.0
        
        state = next_state
        episode_reward += reward
        
        if len(b_states) == BATCH_SIZE or done:
            batch_s = torch.FloatTensor(np.array(b_states)).to(DEVICE)
            batch_a_oh = torch.FloatTensor(np.array(b_actions_one_hot)).to(DEVICE)
            batch_a_idx = torch.LongTensor(np.array(b_actions_idx)).to(DEVICE)
            batch_r = torch.FloatTensor(np.array(b_rewards)).to(DEVICE)
            batch_ns = torch.FloatTensor(np.array(b_next_states)).to(DEVICE)
            batch_d = torch.FloatTensor(np.array(b_dones)).to(DEVICE)
            
            # 3 öğretmeni aynı anda çalıştır
            em_loss, dist_loss, agent_loss = train_i2a_step((batch_s, batch_a_oh, batch_a_idx, batch_r, batch_ns, batch_d))
            writer.add_scalar("Loss/EM_Total", em_loss, total_steps)
            writer.add_scalar("Loss/Distillation", dist_loss, total_steps)
            writer.add_scalar("Loss/Agent_A2C", agent_loss, total_steps)
            
            b_states, b_actions_one_hot, b_actions_idx, b_rewards, b_next_states, b_dones = [], [], [], [], [], []

    rewards_history.append(episode_reward)
    recent_avg = np.mean(rewards_history[-100:])

    writer.add_scalar("Reward/Episode_Score", episode_reward, episode)
    writer.add_scalar("Reward/Average_100", recent_avg, episode)
    
    if recent_avg > best_reward and episode > 50:
        best_reward = recent_avg
        torch.save(i2a_agent.state_dict(), "models/i2a_best_breakout.pth")
        print(f"YENİ REKOR. Ortalama Skor: {best_reward:.1f} | Model Kaydedildi (Bölüm: {episode})")
        
    if episode % 10 == 0:
        print(f"Bölüm: {episode} | Skor: {episode_reward:.1f} | Ajan Kaybı: {agent_loss:.4f} | EM: {em_loss:.4f} | Dist: {dist_loss:.4f}")

env.close()