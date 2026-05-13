import gymnasium as gym
import numpy as np
import collections

class PseudoCountRewardWrapper(gym.Wrapper):
    def __init__(self, env, hash_function=None, reward_scale=1.0):
        super(PseudoCountRewardWrapper, self).__init__(env)
        
        # Lapan'ın Yuvarlama Fonksiyonu: Sonsuz uzayı sayılabilir kutulara bölmek
        if hash_function is None:
            self.hash_function = lambda obs: tuple(np.round(obs, 3))
        else:
            self.hash_function = hash_function
            
        self.reward_scale = reward_scale
        self.counts = collections.Counter() # Gidilen yerlerin çetelesi tutulur

    def _count_observation(self, obs):
        """Durumu hash'le, sayacı artır ve Merak Puanını (1/sqrt(N)) hesapla"""
        h = self.hash_function(obs)
        self.counts[h] += 1
        return np.sqrt(1 / self.counts[h]) #

    def step(self, action):
        """Çevreden gelen gerçek ödüle (extrinsic) merak ödülünü (intrinsic) ekle"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        extra_reward = self._count_observation(obs)
        total_reward = reward + (self.reward_scale * extra_reward) #
        
        return obs, total_reward, terminated, truncated, info