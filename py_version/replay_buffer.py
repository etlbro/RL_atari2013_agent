import random
from collections import deque, namedtuple

Frame = namedtuple('frame',('state','action','reward','next_state', 'ended'))

class Replay_buffer:
    def __init__(self,cap):
        self.memory = deque(maxlen=cap)

    #add new fram to buffer
    def push(self, *args):
        self.memory.append(Frame(*args))    

    #sample X frames from buffer
    def sample(self,size):
        return random.sample(self.memory,size)

    #allows len to work on this class in other files
    def len(self):
        return len(self.memory)
















