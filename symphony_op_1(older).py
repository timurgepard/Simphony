import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math

explore_noise = 0.2 #kickstarter during exploration
stall_penalty = 0.03 #moving is life, stalling is dangerous
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_back = 3


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
    
class FourierSeries(nn.Module):
    def __init__(self, f_in, hidden_dim, f_out):
        super().__init__()


        self.fft = nn.Sequential(
            nn.Linear(f_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, f_out)
        )


    def forward(self, x):
        return self.fft(x)
        



# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device

        self.net = nn.Sequential(
            FourierSeries(state_dim, hidden_dim, action_dim),
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
                self.eps = explore_noise * self.max_action * (math.cos(self.x_coor) + 1)
                self.lim = 2.5*self.eps
                self.x_coor += 7e-5
        return True


    def forward(self, state, mean=False):
        x = self.max_action*self.net(state)
        if mean: return x
        if explore_noise and self.accuracy(): x += (self.eps*torch.randn_like(x)).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)


        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()
    

        self.qA = FourierSeries(state_dim+action_dim, hidden_dim, 1)
        self.qB = FourierSeries(state_dim+action_dim, hidden_dim, 1)
        self.qC = FourierSeries(state_dim+action_dim, hidden_dim, 1)

       
    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        qA, qB, qC= self.qA(x), self.qB(x), self.qC(x)
        if not united: return (qA, qB, qC)
        stack = torch.stack([qA, qB, qC], dim=-1)
        return torch.min(stack, dim=-1).values


# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0):

        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action=max_action).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)

        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_old_policy = 0.0


    def select_action(self, states):
        with torch.no_grad():
            states = np.array(states)
            state = torch.FloatTensor(states).reshape(-1,self.state_dim).to(self.device)
            action = self.actor(state, mean=False)
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
        actor_loss = -ReHaE(q_new_policy - self.q_old_policy)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        with torch.no_grad():
            self.q_old_policy = q_new_policy.mean().detach()

        return q_new_policy.mean()



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, fade_factor=7.0, stall_penalty=0.03):
        self.capacity, self.length, self.device = 500000, 0, device
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



    def add(self, state, action, reward, next_state, done):
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)

        idx = self.length-1

        #moving is life, stalling is dangerous
        delta = np.mean(np.abs(next_state - state))
        reward += self.stall_penalty*(delta - math.log(max(1.0/(delta+1e-6), 1e-3)))

        self.states[idx,:] = torch.FloatTensor(state).to(self.device)
        self.actions[idx,:] = torch.FloatTensor(action).to(self.device)
        self.rewards[idx,:] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx,:] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx,:] = torch.FloatTensor([done]).to(self.device)

        self.batch_size = min(max(128,self.length//500), 1024)


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
