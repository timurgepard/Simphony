import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import math
import copy
import random


# random seeds
r1, r2, r3 = 830143436, 167430301, 2193498338 #r = random.randint(0,2**32-1)
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)

#lifted and squashed Tanh instead of Sigmoid
class StdTanh(nn.Module):
    def forward(self, x):
        return 0.5*(torch.tanh(x)+1.0)

class ReSine(nn.Module):
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)

class ReNU(nn.Module):
    def forward(self, x):
        return (torch.tanh(x)+1.0) * x/2

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)



class Input(nn.Module):
    def __init__(self, f_in, hidden_dim):
        super().__init__()

        #self.nets = nn.ModuleList([nn.Linear(f_in, hidden_dim//4) for _ in range(4)])

        self.out = nn.Sequential(
            nn.Linear(f_in, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        #xs = [net(x) for net in self.nets]
        #return self.out(torch.cat(xs, dim=-1))
        return self.out(x)


class RectFourier(nn.Module):
    def __init__(self, hidden_dim, f_out):
        super().__init__()

        self.fft = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ReSine(),
            nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x):
        return self.fft(x)


class FourierSeries(nn.Module):
    def __init__(self, hidden_dim, f_out):
        super().__init__()

        self.fft1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.Linear(hidden_dim, f_out)
        )

        self.fft2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Cosine(),
            nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x):
        return self.fft1(x) + self.fft2(x)
    

class DeepNeuralNetwork(nn.Module):
    def __init__(self, f_in, hidden_dim, f_out):
        super().__init__()

        self.input = nn.Linear(f_in, hidden_dim)
        self.net = FourierSeries(hidden_dim, f_out)

    def forward(self, x):
        x = self.input(x)
        return self.net(x)
    

#Rectified Hubber Error Loss Function
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function
def ReHaE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0, ou_process = False):
        super(Actor, self).__init__()

        self.input = Input(state_dim, hidden_dim)

        self.net = nn.Sequential(
            RectFourier(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action
        self.disturbance = Disturbance(action_dim, max_action, device, ou_process)
    

    def forward(self, state):
        x = self.input(state)
        x = self.max_action*self.net(x)
        x += (0.07*torch.randn_like(x)).clamp(-0.175, 0.175)
        return x.clamp(-self.max_action, self.max_action)


    def action(self, state):
        with torch.no_grad():
            x = self.input(state)
            x = self.max_action*self.net(x)
            x += self.disturbance.generate(x)
        return x.clamp(-self.max_action, self.max_action)




# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()
        
        self.input = Input(state_dim+action_dim, hidden_dim)

        qA = RectFourier(hidden_dim, 1)
        qB = RectFourier(hidden_dim, 1)
        qC = RectFourier(hidden_dim, 1)

        s2 = RectFourier(hidden_dim, 1)

        self.nets = nn.ModuleList([qA, qB, qC, s2])


    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        x = self.input(x)
        xs = [net(x) for net in self.nets]
        if not united: return xs
        stack = torch.stack(xs[:3], dim=-1)
        return torch.min(stack, dim=-1).values, xs[3]



# Define the environment network
class Environment(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Environment, self).__init__()

        next_state = DeepNeuralNetwork(state_dim+action_dim, hidden_dim, state_dim)

        reward = DeepNeuralNetwork(state_dim+action_dim, hidden_dim, 1)

        done = DeepNeuralNetwork(state_dim+action_dim, hidden_dim, 1)

        self.nets = nn.ModuleList([next_state, reward, done])


    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return [net(x) for net in self.nets]

    def step(self, state, action):
        with torch.no_grad():
            x = torch.cat([state, action], -1)
            xs = [net(x).cpu().data.numpy().flatten() for net in self.nets]
        return xs[0], xs[1], xs[2]




class Dreamer(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device):

        self.env = Environment(state_dim, action_dim, hidden_dim).to(device)

        self.env_optimizer = optim.Adam(self.env.parameters(), lr=7e-4)

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = None

    def train(self, batch):
        state, action, reward, next_state, done = batch
        return self.env_update(state, action, reward, next_state, done)


    def env_update(self, state, action, reward, next_state, done): 

        out = self.env(state, action)
        env_loss = ReHE(next_state - out[0]) + ReHE(reward - out[1]) + ReHE(done - out[2])


        self.env_optimizer.zero_grad()
        env_loss.backward()
        self.env_optimizer.step()

        self.loss = env_loss.detach().mean()

        return self.loss.item()


    def init(self, state):
        self.state = state


    def step(self, action):
        state = torch.FloatTensor(self.state).reshape(-1,self.state_dim).to(self.device)
        action = torch.FloatTensor(action).reshape(-1,self.action_dim).to(self.device)
        next_state, reward, done =  self.env.step(state, action)
        done = True if done>0.75 else False
        self.state = next_state
        return next_state, reward, done, self.loss, None



# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, ou_process=False):

        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action, ou_process).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)

        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0


    def select_action(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
        action = self.actor.action(state)
        return action.cpu().data.numpy().flatten()



    def train(self, batch):
        state, action, reward, next_state, done = batch
        next_action = self.critic_update(state, action, reward, next_state, done)
        return self.actor_update(state, next_state, next_action)



    def critic_update(self, state, action, reward, next_state, done): 

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)

            next_action = self.actor(next_state)
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward +  (1-done) * 0.99 * q_next_target
            s2_value =  3e-3 * (3e-3 * torch.var(reward) +  (1-done) * 0.99 * s2_next_target)
            
        out = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - out[0]) + ReHE(q_value - out[1]) + ReHE(q_value - out[2]) + ReHE(s2_value - out[3])


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return next_action


    def actor_update(self, state, next_state, next_action):
        with torch.no_grad():
            next_q_old_policy, next_s2_old_policy = self.critic(next_state, next_action, united=True)
            next_q_old_policy, next_s2_old_policy = next_q_old_policy.mean(), next_s2_old_policy.mean()

        action = self.actor(state)
        q_new_policy, s2_new_policy = self.critic(state, action, united=True)
        actor_loss = -ReHaE(q_new_policy - self.q_old_policy) -ReHaE(s2_new_policy - self.s2_old_policy)


        next_action = self.actor(next_state)
        next_q_new_policy, next_s2_new_policy = self.critic(next_state, next_action, united=True)
        actor_loss += -ReHaE(next_q_new_policy - next_q_old_policy) -ReHaE(next_s2_new_policy - next_s2_old_policy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        self.q_old_policy = q_new_policy.detach().mean()
        self.s2_old_policy = s2_new_policy.detach().mean()

        return actor_loss



#Replay Buffer with Fading Memories
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=3.0, stall_penalty=0.03):
        capacity_dict = {"short": 100000, "medium": 300000, "full": 500000}
        self.capacity, self.length, self.device = capacity_dict[capacity], 0, device
        self.batch_size = min(max(128, self.length//500), 1024) #in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs, self.step = [], np.array([]), np.array([]), 0
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)

        self.dataset = deque(maxlen=3000000)

    def store(self, state, action, reward, next_state, done):
        self.dataset.append([state, action, reward, next_state, done])
        self.dataset_length = len(self.dataset)
        self.dataset_idx = np.arange(0, self.dataset_length)
        


    def add(self, state, action, reward, next_state, done):
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)

        idx = self.length-1

        #moving is life, stalling is dangerous
        delta = np.mean(np.abs(next_state - state)).clip(1e-1, 10.0)
        reward -= self.stall_penalty*math.log10(1.0/delta)

        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)


        self.states[idx,:] = torch.FloatTensor(state).to(self.device)
        self.actions[idx,:] = torch.FloatTensor(action).to(self.device)
        self.rewards[idx,:] = torch.FloatTensor(np.array([reward])).to(self.device)
        self.next_states[idx,:] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx,:] = torch.FloatTensor(np.array([done])).to(self.device)

        self.batch_size = min(max(128,self.length//500), 1024)

        self.step += 1


    def generate_probs(self, uniform=False):
        def fade(norm_index): return np.tanh(self.fade_factor*norm_index**2) # linear / -> non-linear _/‾
        if uniform: return np.ones(self.length)/self.length
        if self.step>self.capacity: return self.probs
        weights = 1e-7*(fade(self.indexes/self.length))# weights are based solely on the history, highly squashed
        self.probs = weights/np.sum(weights)
        return self.probs


    def sample(self, uniform=False):
        indices = self.random.choice(self.indexes, p=self.generate_probs(uniform), size=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def batch(self, uniform=False):
        def fade(norm_index): return np.tanh(self.fade_factor*norm_index**2)

        if uniform:
            probs = np.ones(self.dataset_length)/self.dataset_length
        else:
            weights = 1e-7*(fade(self.dataset_idx/self.dataset_length))# weights are based solely on the history, highly squashed
            probs = weights/np.sum(weights)


        batch_indices = self.random.choice(self.dataset_idx, p=probs, size=self.batch_size)
        batch = [self.dataset[indx-1] for indx in batch_indices]
        states, actions, rewards, next_states, dones = map(np.vstack, zip(*batch))

        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )


    def __len__(self):
        return self.length




# OU Varying Signal without Gaussian Noise==============================================================


class Disturbance:
    def __init__(self, action_dim, max_action, device, ou_process=False):
        self.eps_coor = 0.0
        self.x_coor = 0.0
        self.ou_process = ou_process
        self.scale = 0.15
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        self.reset()

    def reset(self):

        self.state = torch.zeros(self.action_dim).to(self.device)

        self.phase_act = self.action_dim*torch.rand(self.action_dim).to(self.device)
        self.phase_amp = self.action_dim*torch.rand(self.action_dim).to(self.device)
        self.phase_freq = self.action_dim*torch.rand(self.action_dim).to(self.device)


    def disturbance(self):
        with torch.no_grad():
            self.amplitude = 1.0 + torch.sin(self.x_coor + self.phase_amp)+1.0
            self.frequency = 1.0 + torch.sin(self.x_coor + self.phase_freq)+1.0
            x = torch.sin(self.frequency*self.x_coor + self.phase_act)/self.amplitude
            self.x_coor += 0.01
        return x

    def generate(self, x):
        def descent(x): return 1.07 - math.tanh( x-1.5)
        with torch.no_grad():
            eps = descent(self.eps_coor)
            self.eps_coor += 3e-5
            noise = self.scale*torch.randn_like(x).clamp(-2.5, 2.5)

            if self.ou_process:
                ou_bias = self.state
                noise -= self.scale*ou_bias
                self.state = ou_bias + noise
            else:
                self.state = noise
        return eps*self.max_action*self.state
    

