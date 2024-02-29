#This is the second version of Symphony algorithm.
#The learning speed is slowed down by approx 20% to learn (Bellemare et al) and increase (something new) Q variance.
#Advantage in Q Variance's empirically has gradient with relation to actions.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
import random



# random seeds
r1, r2, r3 = 830143436, 167430301, 2193498338 #r = random.randint(0,2**32-1)
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)


#Rectified Hubber Error Loss Function
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function
def ReHaE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class ReSine(nn.Module):
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)



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
        


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0, burst=False, tr_noise=True, ou_process=False):
        super(Actor, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )

        self.net = nn.Sequential(
            RectFourier(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        self.disturbance = Disturbance(action_dim, max_action, device, burst, tr_noise, ou_process)
    

    def forward(self, state):
        x = self.input(state)
        x = self.max_action*self.net(x)
        return x.clamp(-self.max_action, self.max_action)


    def action(self, state, mean=False):
        with torch.no_grad():
            x = self.input(state)
            x = self.max_action*self.net(x)
            if not mean: x += self.disturbance.generate()
        return x.clamp(-self.max_action, self.max_action)


# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()
        
        self.input = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.Tanh()
        )

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
        qmin = torch.min(torch.stack(xs[:3], dim=-1), dim=-1).values
        return qmin, xs[3]




# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action, capacity, fade_factor, stall_penalty, burst=False, tr_noise=True, ou_process=False):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, capacity, device, fade_factor, stall_penalty)

        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action, burst, tr_noise, ou_process).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)

        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_old_policy, self.s2_old_policy = 0.0, 0.0
        self.next_q_old_policy, self.next_s2_old_policy = 0.0, 0.0
        self.tr_step = 0


    def select_action(self, state, mean=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
            state = self.replay_buffer.normalize(state)
            action = self.actor.action(state, mean=mean)
        return action.cpu().data.numpy().flatten()


    def train(self, uniform=False):
        self.tr_step += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(uniform)
        self.critic_update(state, action, reward, next_state, done)
        return self.actor_update(state)


    def critic_update(self, state, action, reward, next_state, done):

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)

            next_action = self.actor(next_state)
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward + (1-done) * 0.99 * q_next_target
            s2_value =  3e-3 * (3e-3 * torch.var(reward) + (1-done) * 0.99 * s2_next_target) #reduced objective to learn Bellman's sum of dumped variance

        out = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - out[0]) + ReHE(q_value - out[1]) + ReHE(q_value - out[2]) + ReHE(s2_value - out[3])
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def actor_update(self, state):

        
        action = self.actor(state)
        q_new_policy, s2_new_policy = self.critic(state, action, united=True)
        actor_loss = -ReHaE(q_new_policy - self.q_old_policy) -ReHaE(s2_new_policy - self.s2_old_policy)

        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        with torch.no_grad():
            self.q_old_policy = q_new_policy.detach().mean()
            self.s2_old_policy = s2_new_policy.detach().mean()
        
        return actor_loss



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
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

        self.raw = True


    def find_min_max(self):
        self.min_values = torch.min(self.states, dim=0).values
        self.max_values = torch.max(self.states, dim=0).values

        self.min_values[torch.isinf(self.min_values)] = -1e+6
        self.max_values[torch.isinf(self.max_values)] = +1e+6

        self.min_values = 2.0*(torch.floor(10.0*self.min_values)/10.0).reshape(1, -1).to(self.device)
        self.max_values = 2.0*(torch.ceil(10.0*self.max_values)/10.0).reshape(1, -1).to(self.device)

        self.raw = False



    def normalize(self, state):
        if self.raw: return state
        state = state.clamp(-1e+6, +1e+6)
        state = 4.0 * (state - self.min_values) / ((self.max_values - self.min_values)) - 2.0
        state[torch.isnan(state)] = 0.0
        return state


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
        self.rewards[idx,:] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx,:] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx,:] = torch.FloatTensor([done]).to(self.device)

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
            self.normalize(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self.normalize(self.next_states[indices]),
            self.dones[indices]
        )




    def __len__(self):
        return self.length
    


# NOISES with cosine decrease==============================================================

class Disturbance:
    def __init__(self, action_dim, max_action, device, burst=False, tr_noise=True, ou_process=False):
        self.eps_coor = 0.0
        self.x_coor = 0.0
        self.scale = 1.0*max_action if burst else 0.15*max_action
        self.tr_noise = 0.07 if tr_noise else 0.0
        self.action_dim = action_dim
        self.device = device
        self.ou_process = ou_process

        self.reset()

    def reset(self):

        self.dist_state = torch.zeros(self.action_dim).to(self.device)

        self.phase_act = self.action_dim*torch.rand(self.action_dim).to(self.device)
        self.phase_amp = self.action_dim*torch.rand(self.action_dim).to(self.device)
        self.phase_freq = self.action_dim*torch.rand(self.action_dim).to(self.device)

    def disturbance(self):
        with torch.no_grad():
            self.amplitude = 1.0 + torch.sin(self.x_coor + self.phase_amp)+1.0
            self.frequency = 1.0 + torch.sin(self.x_coor + self.phase_freq)+1.0
            x = torch.sin(self.frequency*self.x_coor + self.phase_act)/self.amplitude
            self.x_coor += 0.1
        return x

    def generate(self):
        with torch.no_grad():
            eps = (1.0 - math.tanh(self.eps_coor)) + self.tr_noise
            #lim = 2.5*eps
            self.eps_coor += 3.07e-5
            noise = self.scale * self.disturbance()
            #if self.ou_process:
            ou_bias = self.dist_state
            noise -= self.scale * ou_bias
            self.dist_state = ou_bias + noise
            #else:
                #self.dist_state = noise
        return eps*self.dist_state

