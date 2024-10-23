
'''
References:
Sutton (1996), Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding
Sutton, http://incompleteideas.net/tiles/tiles3.html
Sutton & Barto (2018), http://incompleteideas.net/book/the-book-2nd.html

Tested Params (not optimised):
Acrobot-v1: alpha=0.5, lmda=0.9, gamma=1, epsilon=0.0, max_size=65536, nb_tilings=32, tile_interval=3
MountainCar-v0: alpha=1, lmda=0.9, gamma=1, epsilon=0.0, max_size=65536, nb_tilings=32, tile_interval=10
CartPole-v1: alpha=1, lmda=0.9, gamma=1, epsilon=0.0, max_size=65536, nb_tilings=10, tile_interval=10
'''

import numpy as np

class TileCodingSARSA():
    '''Sarsa(λ) with binary features and linear function approximation for estimating wᵀx ≈ qπ or q*'''
    def __init__(self, env, alpha=1, lmda=0.9, gamma=1, epsilon=0.0, max_size=65536, nb_tilings=10, tile_interval=10):
        '''
        Params:
            env: Gym environment (Tested: Acrobot-v1, CartPole-v1, MountainCar-v0)
            alpha: Learning rate 
            lmbda: λ, Eligibility trace decay parameter
            gamma: Discount factor
            epsilon: e-greedy value (exploration versus exploitation)
            max_size: Index hash table size
            nb_tilings: Number of tiling layers
            tile_interval: tile_interval x tile_interval grids per tile
        '''
        self.env = env
        self.alpha = alpha / nb_tilings
        self.lmda = lmda
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_size = max_size
        self.nb_tilings = nb_tilings
        self.tile_interval = tile_interval
        self.iht = IHT(max_size) # Index hash table
        self.w = np.zeros(max_size) # Weights
        self.z = np.zeros(max_size) # Eligibility trace

        # For CartPole-v1 environment where the observation.high and observation.low are inf which messes up with the state normalisations (might need to update for non-tested environments)
        self.env.observation_space.high = np.array([2.5 if self.env.observation_space.high[s] > 100 else self.env.observation_space.high[s] for s in range(self.env.observation_space.shape[0])])
        self.env.observation_space.low = np.array([-2.5 if self.env.observation_space.low[s] < -100 else self.env.observation_space.low[s] for s in range(self.env.observation_space.shape[0])])
        
        assert isinstance(env.action_space, Discrete), "This example only works for envs with discrete action spaces."

    def my_tiles(self, state, action):
        '''Returns which tile indices are active given the current state and action'''
        # Normalises state values between 0-1 and spread across the tile intervals
        state_list = np.array([((state[s] - self.env.observation_space.low[s]) / (self.env.observation_space.high[s] - self.env.observation_space.low[s])) for s in range(self.env.observation_space.shape[0])])      
        state_list *= self.tile_interval
        # Rounded to 3dp to save IHT space (not really necessary for most environments where the state space isn't too large)
        state_list = np.around(state_list, decimals=3)
        return tiles(self.iht, self.nb_tilings, state_list, [action])

    def epilson_greedy(self, state):
        '''Returns, with probability epsilon, a random action or, with probability 1-epsilon, computes an action for which the Q-value is largest, resolving ties randomly'''
        if np.random.rand() < self.epsilon:
            # Exploration
            return self.env.action_space.sample()
        else:
            # Exploitation, calculate the Q-values for each action by summing the weights of the active tiles
            q_values = np.array([self.evaluate(state, action) for action in range(self.env.action_space.n)])
            return self.argmax(q_values)
        
    def argmax(self, arr):
        '''Returns argmax, resolving ties randomly - np.argmax() always picks first max'''
        return np.random.choice(np.flatnonzero(arr == np.max(arr)))

    def evaluate(self, state, action):
        '''Returns the Q-value for a given action by summing the weights associated with the active tiles (state)'''
        tile_indices = self.my_tiles(state, action)
        return np.sum(self.w[tile_indices])
    
    def train(self, state, action, reward, next_state, next_action):
        '''Weight update using the SARSA(λ) update rule'''
        tile_indices = self.my_tiles(state, action)
        current_value = self.evaluate(state, action)
        next_value = self.evaluate(next_state, next_action)
        
        target = reward + self.gamma * next_value
        
        #self.z[tile_indices] += 1 # Accumulating trace
        self.z[tile_indices] = 1 # Replacing trace
        self.w += self.alpha * (target - current_value) * self.z # Weight update
        self.z *= self.gamma * self.lmda # Eligiblity trace decay

    def reset_z(self):
        '''Resets the eligibility trace (at the start of every episode)'''
        self.z = np.zeros(self.max_size)

'''
Tile Coding Software version 3.0beta
by Rich Sutton
'''
basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

'''
Tile Coding Software version 3.0beta
by Rich Sutton
'''

if __name__ == "__main__":
    '''Example use: Import model and run experiment loop in a given environment'''
    import numpy as np
    import matplotlib.pyplot as plt
    import gymnasium as gym
    from gymnasium.spaces import Discrete
    from tqdm import tqdm

    # Basic Experiment Loop, State->Action->Reward->Update Model->Next State->Next Action->...
    def experiment(env_name, n_episodes=500):
        '''Runs experiment given environment and model -> plots a learning curve and returns the model'''

        env = gym.make(env_name)
        # Params for the model
        model = TileCodingSARSA(env, alpha=1, lmda=0.9, gamma=1, epsilon=0.0, max_size=65536, nb_tilings=32, tile_interval=10) 

        learning_curve = []
        for _ in tqdm(range(n_episodes)):
            rewards = 0
            model.reset_z() # Remember to reset eligibilty trace at the start of each episode
            state = env.reset()
            state = state[0] # First state
            action = model.epilson_greedy(state) # Find first action based on e-greedy
            terminated = False
            truncated = False
            while not (terminated or truncated):
                next_state, reward, terminated, truncated, _ = env.step(action) # Take the action to get the next state + reward
                next_action = model.epilson_greedy(next_state) # Find the next action given the next state
                model.train(state, action, reward, next_state, next_action) # SARSA(λ) update rule
                state = next_state
                action = next_action
                rewards += reward
            learning_curve.append(rewards)

        plt.plot(range(n_episodes), learning_curve[:])
        plt.title(f'Learning Curve, Best Reward Over {n_episodes} Episodes: {np.max(learning_curve)}')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards per episode')
        plt.show()
        return model
    
    model = experiment(env_name='MountainCar-v0', n_episodes=500)