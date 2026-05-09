import numpy as np
from collections import deque # performans için collections.deque kullanıyoruz. Kapasite dolduğunda yeni bir anı eklenirse, Python en eski anıyı O(1) hızında (anında) siler.
import random

class ReplayBuffer:

    def __init__(self, capacity=1000000):

        # deque (Double Ended Queue): Kapasite dolduğunda en eski anıları otomatik siler
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        # Günlükte şu an kaç anı biriktiğini döndürür
        return len(self.buffer)
    
    def push(self, state, action, reward, next_state, done):
        # Ajanın o anki deneyimini/Transition tuple olarak ekliyoruz 
        # Mujoco'da boyut uyumsuzluğu olmaması için her şeyi numpy array/float formatında tutuyoruz
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Günlükten rastgele (batch_size kadar) anı çekeriz
        # Rastgelelik önemlidir. Ajanın hep aynı tip veriyi üst üste görüp ezberlemesini önler
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        return np.concatenate(state), action, reward, np.concatenate(next_state), done