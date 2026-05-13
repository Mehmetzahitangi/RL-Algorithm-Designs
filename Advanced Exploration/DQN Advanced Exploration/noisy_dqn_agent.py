import torch
import torch.nn as nn
import numpy as np
import random
import copy

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
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

class NoisyDQNAgent:
    def __init__(self, net, lr=1e-3, gamma=0.99, device="cpu"):
        self.net = net.to(device)
        self.target_net = copy.deepcopy(self.net).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.device = device

    def select_action(self, state):
        """EPSILON YOK! Ajan her zaman en inandığı şeyi yapar (Ama inancı gürültülüdür)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.net(state_t)
            action = q_values.max(dim=1)[1].item()
        return action

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1. Mevcut Q Değerleri
        q_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # 2. Hedef Q Değerleri (Gelecek)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_q = rewards + self.gamma * next_q_values * (1 - dones)
            
        # 3. Kayıp ve Optimizasyon
        loss = nn.MSELoss()(q_values, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. LAPAN'IN KRİTİK KURALI: Her eğitim adımında gürültüyü yenile
        # Ajan ağırlıkları güncelledikten sonra yeni bir "Merak" profiline bürünür.
        self.net.sample_noise()
        self.target_net.sample_noise()
        
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.net.state_dict())