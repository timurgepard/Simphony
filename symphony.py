import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import copy
import pickle
import time
import random
from collections import deque
import math




env = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
env_test = gym.make('BipedalWalkerHardcore-v3', render_mode="human")
variable_steps = True
limit_steps = 30
start_time = 2000


# 4 random seeds
#r1 = random.randint(0,2**32-1)
#r2 = random.randint(0,2**32-1)
#r3 = random.randint(0,2**32-1)
r1, r2, r3 = 3684114277, 3176436971, 2662091060
r1, r2, r3 = 1375924064, 4183630884, 3128176189
r1, r2, r3 = 1495103356, 3007657725, 417911830
r1, r2, r3 = 830143436, 167430301, 2193498338
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)
random.seed(r3)


#used to create random seeds in Actor -> less dependendance on the specific neural network random seed.
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


#Rectified Hubber Error Loss Function, stabilizes the learning speed
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function, stabilizes the learning speed
def ReHAE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)

def harmonize(state):
    sin_part = np.sin(state)
    cos_part = np.cos(state)
    sin_cos = sin_part*cos_part
    return np.concatenate((state, sin_part, cos_part, sin_cos), axis=-1)

#testing model
def testing(env, algo, limit_steps, test_episodes):
    if test_episodes<1: return
    episode_return = []

    for test_episode in range(test_episodes):
        state = env.reset()[0]
        state = harmonize(state)
        rewards = []

        for steps in range(1,limit_steps+1):
            action = algo.select_action(state, mean=True)
            next_state, reward, done, info , _ = env.step(action)
            next_state = harmonize(next_state)
            state = next_state
            rewards.append(reward)
            if done: break

        episode_return.append(np.sum(rewards))

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode}:, Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}")

        if test_episodes==1000 and validate_return>=300: print("Average of 100 trials = 300 !!!CONGRATULATIONS!!!")


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
         )
        
        self.max_action = torch.mean(max_action).item()
        self.eps = 0.3
        self.lim = 2.5*self.eps
        self.x_coor = 0.0

    def accuracy(self):
        if self.eps<0.01: return False
        if self.eps>0.1:
            self.eps = 0.3 * self.max_action * math.exp(-self.x_coor)
            self.lim = 2.5*self.eps
            self.x_coor += 3e-5
        else:
            self.eps = 0.1
            self.lim = 2.5*self.eps
        return True
        

    def forward(self, state, mean=False, extra_noise=False):
        x = self.max_action*self.net(state)
        if mean: return x
        if extra_noise and self.accuracy(): return x + (self.eps*self.max_action*torch.randn_like(x)).clamp(-self.lim, self.lim)
        x += (0.2*self.max_action*torch.randn_like(x)).clamp(-0.5, 0.5)
        return x.clamp(-1.0, 1.0)

        
        
# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()

        self.netA = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.netB = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.netC = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        qA, qB, qC = self.netA(x), self.netB(x), self.netC(x)
        if not united: return (qA, qB, qC)
        stack = torch.stack([qA, qB, qC], dim=-1)
        return torch.min(stack, dim=-1).values




# we use cache to append transitions, and then update buffer to collect Q Returns, purging cache at the episode end.
class ReplayBuffer:
    def __init__(self, device, capacity=1000*2560):
        self.buffer, self.capacity, self.length =  deque(maxlen=capacity), capacity, 0 #buffer is prioritised limited memory
        self.device = device
        self.batch_size = min(max(32, self.length//300), 2560) #in order for sample to describe population
        self.random = np.random.default_rng()

    
    def add(self, transition):
        self.buffer.append(transition)
        self.length = len(self.buffer)
        self.batch_size = min(max(32, self.length//300), 2560)

    def sample(self):
        indexes = np.array(list(range(self.length)))
        weights = 0.001*(indexes/self.length)
        probs = weights/np.sum(weights)

        batch_indices = self.random.choice(indexes, p=probs, size=self.batch_size)
        batch = [self.buffer[indx-1] for indx in batch_indices]
        states, actions, rewards, next_states, dones = map(np.vstack, zip(*batch))

        
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)


# Define the actor-critic agent
class uDDPG(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0):

        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action=max_action).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)

        self.max_action = max_action
        self.device = device
        self.action_dim = action_dim


    def select_action(self, state, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1,state.shape[-1]).to(self.device)
            action = self.actor(state, mean=mean, extra_noise=True)
        return action.cpu().data.numpy().flatten()


    def train(self, batch):
        state, action, reward, next_state, done = batch
        self.critic_update(state, action, reward, next_state, done)
        self.actor_update(state)


    def critic_update(self, state, action, reward, next_state, done): 

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)

            next_action = self.actor(next_state, mean=False)
            q_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward +  (1-done) * 0.99 * q_next_target

        qA, qB, qC = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - qA) + ReHE(q_value - qB) + ReHE(q_value - qC)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def actor_update(self, state):

        action = self.actor(state, mean=True)
        q_new_policy = self.critic(state, action, united=True)
        actor_loss = -q_new_policy
        actor_loss = ReHAE(actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count())
        
print(device)



state_dim = 4*env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

hidden_dim = 256

print('action space high', env.action_space.high)

max_action = torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else 1.0
replay_buffer = ReplayBuffer(device)
algo = uDDPG(state_dim, action_dim, hidden_dim, device, max_action)

num_episodes, total_rewards, total_steps, test_rewards, policy_training = 1000000, [], [], [], False


try:
    print("loading buffer...")
    with open('replay_buffer', 'rb') as file:
        dict = pickle.load(file)
        replay_buffer = dict['buffer']
        algo.actor.eps = dict['eps']
        algo.actor.x_coor = dict['x_coor']
        limit_steps = dict['limit_steps']
        total_steps = dict['total_steps']
        if len(replay_buffer)>=start_time and not policy_training: policy_training = True
    print('buffer loaded, buffer length', len(replay_buffer))

except:
    print("problem during loading buffer")

#load existing models

try:
    print("loading models...")
    algo.actor.load_state_dict(torch.load('actor_model.pt'))
    algo.critic.load_state_dict(torch.load('critic_model.pt'))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt'))

    print('models loaded')

    testing(env_test, algo, limit_steps, 10)

except:
    print("problem during loading models")




for i in range(num_episodes):
    rewards = []
    state = env.reset()[0]
    state = harmonize(state)
   

    #---------------------------1. processor releave --------------------------
    #---------------------2. decreases dependence on random seed: ---------------
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    #-----------prevents often appearance of the same transitions in buffer-------

    #1
    if policy_training: time.sleep(2.0)
    #2
    if not policy_training and len(replay_buffer.buffer)<start_time: algo.actor.apply(init_weights)
    #3

    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)

    for t in range(0, 8):
        next_state, reward, done, info, _ = env.step(action)
        next_state = harmonize(next_state)
        state = next_state
        rewards.append(reward)

    #------------------------------training------------------------------

    for steps in range(1, limit_steps+1):

        if len(replay_buffer.buffer)>=start_time and not policy_training:
            print("started training")
            policy_training = True
            for _ in range(128):
                algo.train(replay_buffer.sample())


        action = algo.select_action(state, mean=True)
        next_state, reward, done, info, _ = env.step(action)
        next_state = harmonize(next_state)
        replay_buffer.add([state, action, reward, next_state, done])
        rewards.append(reward)
            
            
        if policy_training:
            algo.train(replay_buffer.sample())
            algo.train(replay_buffer.sample())
            algo.train(replay_buffer.sample())
                
        
        state = next_state
        if done: break
            

    total_rewards.append(np.sum(rewards))
    average_reward = np.mean(total_rewards[-100:])

    episode_steps = steps
    total_steps.append(episode_steps)
    average_steps = np.mean(total_steps[-100:])
    if policy_training and variable_steps and limit_steps<=2000: limit_steps = int(average_steps) + 5 + int(0.05*average_steps)**2


    print(f"Ep {i}: Rtrn = {total_rewards[i]:.2f} | ep steps = {episode_steps}")



    if policy_training:

        #--------------------saving-------------------------
        if (i>=10 and i%10==0): 
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            #print("saving... len = ", len(replay_buffer), end="")
            with open('replay_buffer', 'wb') as file:
                pickle.dump({'buffer': replay_buffer, 'eps': algo.actor.eps, 'x_coor':algo.actor.x_coor, 'limit_steps':limit_steps, 'total_steps':total_steps}, file)
            #print(" > done")


        #-----------------validation-------------------------

        if total_rewards[i]>=330 or (i>=200 and i%100==0):
            test_episodes = 1000 if total_rewards[i]>=330 else 5
            env_val = env if test_episodes == 1000 else env_test
            print("Validation... ", test_episodes, " epsodes")
            test_rewards = []

            testing(env_val, algo, limit_steps, test_episodes)
                    

        #====================================================
