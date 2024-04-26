import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math
import random
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# random seeds
r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
#r1, r2, r3 = 1216815315,  386391682,  1679869492
#r1, r2, r3 = 3743773989,  91389447,  1496950741
#r1, r2, r3 = 4192644313, 3075238889, 2575656344
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)
random.seed(r3)

class LogFile(object):
    def __init__(self, log_name):
        self.log_name= log_name
    def write(self, text):
        with open(self.log_name, 'a+') as file:
            file.write(text)

log_name = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_file = LogFile(log_name)

#Rectified Hubber Error Loss Function
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function
def ReHaE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)

class ReSine(nn.Module):
    def forward(self, x):
        return F.leaky_relu(torch.sin(x), 0.1)

class Input(nn.Module):
    def __init__(self, f_in, hidden_dim):
        super().__init__()

        inA = nn.Linear(f_in, hidden_dim//3)
        inB = nn.Linear(f_in, hidden_dim//3)
        inC = nn.Linear(f_in, hidden_dim//3)

        self.nets = nn.ModuleList([inA, inB, inC])


    def forward(self, x):
        xs = [net(x) for net in self.nets]
        return torch.cat(xs, dim=-1)



class FourierSeries(nn.Module):
    def __init__(self, f_in, hidden_dim, f_out):
        super().__init__()


        self.fft = nn.Sequential(
            nn.Linear(f_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            ReSine(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, f_out)
        )


    def forward(self, x):
        return self.fft(x)

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device

        self.input = Input(state_dim, hidden_dim)

        self.net = nn.Sequential(
            FourierSeries(hidden_dim, hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        self.scale = 0.2*self.max_action
        self.lim = 2.5*self.scale
        self.eps_coor = 0.0
    
    
    def forward(self, state):
        x = self.input(state)
        x = self.max_action*self.net(x)
        return x

    def action(self, state, mean=False):
        with torch.no_grad():
            x = self.forward(state)
            if mean: return x
            x += (self.scale*torch.randn_like(x)).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)


        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        #self.input = Input(state_dim+action_dim, hidden_dim)

        qA = FourierSeries(state_dim+action_dim, hidden_dim, 3)
        qB = FourierSeries(state_dim+action_dim, hidden_dim, 3)
        qC = FourierSeries(state_dim+action_dim, hidden_dim, 3)

        self.nets = nn.ModuleList([qA, qB, qC])

       
    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        #x = self.input(x)
        xs = [net(x) for net in self.nets]
        if not united: return xs
        return torch.min(torch.cat(xs, dim=-1), dim=-1, keepdim=True).values
        #return torch.min(xs, dim=-1, keepdim=True).values


# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, fade_factor=7.0, lambda_r=0.07):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, fade_factor, lambda_r)

        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action=max_action).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_old_policy = 0.0



    def select_action(self, states, mean=False):
        with torch.no_grad():
            states = np.array(states)
            state = torch.FloatTensor(states).reshape(-1,self.state_dim).to(self.device)
            action = self.actor.action(state, mean=mean)
        return action.cpu().data.numpy().flatten()


    def train(self, tr_per_step=1):
        #Update-To-Data (UTD)
        for _ in range(tr_per_step):
            state, action, reward, next_state, done = self.replay_buffer.sample()
            #Sample Multiple Reuse (SMR)
            for _ in range(3):
                self.critic_update(state, action, reward, next_state, done)
            self.actor_update(state)
            


    def critic_update(self, state, action, reward, next_state, done): 

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)

            next_action = self.actor(next_state)
            q_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward +  (1-done) * 0.99 * q_next_target

        qs = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - qs[0]) + ReHE(q_value - qs[1]) + ReHE(q_value - qs[2])


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        

    def actor_update(self, state):
        action = self.actor(state)
        q_new_policy = self.critic(state, action, united=True)
        actor_loss = -ReHaE(q_new_policy - self.q_old_policy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        with torch.no_grad(): self.q_old_policy = q_new_policy.mean().detach()
            



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, fade_factor=7.0, lambda_r=0.07):
        self.capacity, self.length, self.device = 512000, 0, device
        self.batch_size = min(max(128, self.length//250), 2048) #in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs, self.step = [], np.array([]), np.array([]), 0
        self.fade_factor = fade_factor

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)

        self.alpha_base = lambda_r
        self.rewards_sum = 0
        self.n_sigma = 0
        self.delta_max = 3.0
        self.alpha = lambda_r


    def add(self, state, action, reward, next_state, done):
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)

        idx = self.length-1
        

        #moving is life, stalling is dangerous
        self.rewards_sum += (reward/2)*math.tanh(reward/2)
        self.step += 1

     
        self.alpha = self.alpha_base*(10.0 + self.rewards_sum/self.step)/10.0
        

        delta = np.mean(np.abs(next_state - state)).item()
        if self.step<=1000:
            #self.alpha = self.alpha_base*(10.0+self.rewards_sum/self.step)/10.0
            if delta>self.delta_max: self.delta_max = delta
            if self.step==1000: print('alpha = ', round(self.alpha, 3))

        delta /= self.delta_max
        reward += self.alpha*(0.5*math.tanh(math.log(2.0*delta))+0.5)
        

        self.states[idx,:] = torch.FloatTensor(state).to(self.device)
        self.actions[idx,:] = torch.FloatTensor(action).to(self.device)
        self.rewards[idx,:] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx,:] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx,:] = torch.FloatTensor([done]).to(self.device)

        self.batch_size = min(max(128, self.length//250), 2048)


        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)

        


    def generate_probs(self):
        if self.step>self.capacity: return self.probs
        def fade(norm_index): return np.tanh(self.fade_factor*norm_index**2) # linear / -> non-linear _/‾
        weights = 1e-7*(fade(self.indexes/self.length))# weights are based solely on the history, highly squashed
        self.probs = weights/np.sum(weights)
        return self.probs


    def sample(self):
        indices = self.random.choice(self.indexes, p=self.generate_probs(), size=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


    def __len__(self):
        return self.length
