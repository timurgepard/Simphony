#This is the second version of Symphony algorithm.
#The learning speed is slowed down by approx 20% to learn (Bellemare et al) and increase (something new) Q variance.
#Advantage in Q Variance's empirically has gradient with relation to actions.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math



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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
    
class FourierTransform(nn.Module):
    def __init__(self, seq, hidden_dim, f_out, device):
        super().__init__()

        self.fft = nn.Linear(seq, seq, bias=False)

        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, f_out)
        )

        self.tri = torch.tril(torch.ones((seq, seq))).to(device)
        self.tri_W = self.tri/self.tri.sum(dim=1, keepdim=True)

    def forward(self, x):
        B, T, E = x.shape
        x = x.reshape(B, E, T)
        x = nn.functional.linear(x, self.tri_W[:T,:T].detach() * self.fft.weight[:T,:T], None)
        x = x.reshape(B, T, E)
        return  self.ffw(x)
        



# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, seq, hidden_dim=32, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device


        self.input = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.net = nn.Sequential(
            FourierTransform(seq, hidden_dim, action_dim, device),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()

        self.eps = 1.0
        self.lim = 2.5*self.eps
        self.x_coor = 0.0
    
    def accuracy(self):
        if self.eps<1e-4: return False
        if self.eps>=0.07:
            with torch.no_grad():
                self.eps = 0.15 * self.max_action * (math.cos(self.x_coor) + 1.0)
                self.lim = 2.5*self.eps
                self.x_coor += 3e-5
        return True


    def forward(self, state, mean=False):
        x = self.input(state)
        x = self.max_action*self.net(x)
        if mean: return x
        if self.accuracy(): x += (self.eps*torch.randn_like(x)).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)


# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, seq, hidden_dim=32, critics_average=False):
        super(Critic, self).__init__()
        
        self.input = nn.Linear(state_dim+action_dim, hidden_dim)

        qA = FourierTransform(seq, hidden_dim, action_dim, device)
        qB = FourierTransform(seq, hidden_dim, action_dim, device)
        qC = FourierTransform(seq, hidden_dim, action_dim, device)

        s2 = FourierTransform(seq, hidden_dim, action_dim, device)

        self.nets = nn.ModuleList([qA, qB, qC, s2])

        self.critics_average = critics_average
        

    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        x = self.input(x)
        xs = [net(x).mean(dim=1) for net in self.nets]
        if not united: return xs
        stack = torch.stack(xs[:3], dim=-1)
        min = torch.min(stack, dim=-1).values
        q = min if not self.critics_average else 0.7*min + 0.3*torch.mean(stack, dim=-1)
        return q, xs[3]




# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, seq, hidden_dim, device, max_action=1.0, critics_average=False):

        self.actor = Actor(state_dim, action_dim, device, seq, hidden_dim, max_action=max_action).to(device)

        self.critic = Critic(state_dim, action_dim, device, seq, hidden_dim, critics_average).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)

        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq = seq
        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0


    def select_action(self, states):
        with torch.no_grad():
            states = np.array(states)
            state = torch.FloatTensor(states).reshape(-1, self.seq, self.state_dim).to(self.device)
            action = self.actor(state, mean=False).mean(dim=1)
        return action.cpu().data.numpy().flatten()


    def train(self, batch):
        state, action, reward, next_state, done = batch
        self.critic_update(state, action, reward, next_state, done)
        return self.actor_update(state)


    def critic_update(self, state, action, reward, next_state, done): 

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)

            next_action = self.actor(next_state, mean=True)
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            reward, done = reward[:,-1], done[:,-1]
            q_value = reward +  (1-done) * 0.99 * q_next_target
            s2_value =  1e-3 * (1e-3*torch.var(reward) +  (1-done) * 0.99 * s2_next_target) #reduced objective to learn Bellman's sum of dumped variance

        out = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - out[0]) + ReHE(q_value - out[1]) + ReHE(q_value - out[2]) + ReHE(s2_value - out[3])
        


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        

    def actor_update(self, state):
        action = self.actor(state, mean=True)
        q_new_policy, s2_new_policy = self.critic(state, action, united=True)
        actor_loss = -ReHaE(q_new_policy - self.q_old_policy) -ReHaE(s2_new_policy - self.s2_old_policy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            self.q_old_policy = q_new_policy.mean().detach()
            self.s2_old_policy = s2_new_policy.mean().detach()

        return self.q_old_policy 



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, seq, fade_factor=7.0, stall_penalty=0.03):
        capacity_dict = {"short": 100000, "medium": 300000, "full": 500000}
        self.capacity, self.length, self.device = capacity_dict[capacity], 0, device
        self.batch_size = min(max(128, self.length//500), 256) #in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs, self.step = [], np.array([]), np.array([]), 0
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty
        self.t = seq

        self.states = torch.zeros((self.capacity, seq, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, seq, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, seq, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, seq, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, seq, 1), dtype=torch.float32).to(device)



    def add(self, states, actions, rewards, next_states, dones):
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)

        idx = self.length-1

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        delta = np.mean(np.abs(next_states - states), axis=-1, keepdims=True).clip(1e-3, 10.0)
        rewards -= self.stall_penalty*np.log(1.0/delta)

        self.states[idx, 0:self.t, :] = torch.FloatTensor(states).to(self.device)
        self.actions[idx, 0:self.t, :] = torch.FloatTensor(actions).to(self.device)
        self.rewards[idx, 0:self.t, :] = torch.FloatTensor(rewards).to(self.device)
        self.next_states[idx, 0:self.t, :] = torch.FloatTensor(next_states).to(self.device)
        self.dones[idx, 0:self.t, :] = torch.FloatTensor(dones).to(self.device)

        self.batch_size = min(max(128,self.length//500), 256)


        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)

        self.step += 1


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

