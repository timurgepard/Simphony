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
        

class Input(nn.Module):
    def __init__(self, f_in, hidden_dim):
        super().__init__()

        self.nets = nn.ModuleList([nn.Linear(f_in, hidden_dim//4) for _ in range(4)])

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        xs = [net(x) for net in self.nets]
        return self.net(torch.cat(xs, dim=-1))


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0, ou_process=True):
        super(Actor, self).__init__()

        self.input = Input(state_dim, hidden_dim)

        self.net = nn.Sequential(
            RectFourier(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        self.disturbance = Disturbance(action_dim, max_action, device, ou_process)


    def forward(self, state):
        x = self.input(state)
        x = self.max_action*self.net(x)
        return x


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






# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, ou_process=True):

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
        self.tr_step = 0



    def select_action(self, state, mean=False):
        action = self.actor.action(state, mean)
        return action.cpu().data.numpy().flatten()


    def train(self, batch):
        self.tr_step += 1
        state, action, reward, next_state, done = batch
        self.critic_update(state, action, reward, next_state, done)
        return self.actor_update(state, next_state) if self.tr_step%1==0 else None


    def critic_update(self, state, action, reward, next_state, done): 

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997*target_param.data + 0.003*param)

            next_action = self.actor(next_state)
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward +  (1-done) * 0.99 * q_next_target
            s2_value =  7e-3 * (7e-3 * torch.var(reward) +  (1-done) * 0.99 * s2_next_target) #reduced objective to learn Bellman's sum of dumped variance
            
            next_q_old_policy, next_s2_old_policy = self.critic(next_state, next_action, united=True)
            self.next_q_old_policy, self.next_s2_old_policy = next_q_old_policy.mean(), next_s2_old_policy.mean()

        out = self.critic(state, action, united=False)
        critic_loss = ReHE(q_value - out[0]) + ReHE(q_value - out[1]) + ReHE(q_value - out[2]) + ReHE(s2_value - out[3])
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        

    def actor_update(self, next_state):

        next_action = self.actor(next_state)
        next_q_new_policy, next_s2_new_policy = self.critic(next_state, next_action, united=True)
        actor_loss = -ReHaE(next_q_new_policy - self.next_q_old_policy) -ReHaE(next_s2_new_policy - self.next_s2_old_policy)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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
        self.raw = True

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)

        

    def find_min_max(self):
        self.min_values = torch.min(self.states, dim=0).values
        self.max_values = torch.max(self.states, dim=0).values

        self.min_values[torch.isinf(self.min_values)] = -1e+6
        self.max_values[torch.isinf(self.max_values)] = +1e+6

        self.min_values = 2.0*(torch.floor(10.0*self.min_values)/10.0).reshape(1, -1).to(self.device)
        self.max_values = 2.0*(torch.ceil(10.0*self.max_values)/10.0).reshape(1, -1).to(self.device)

        self.raw = False


    def normalize(self, state):
        if state.ndim == 1: state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
    

# OU with gradually varying signal==============================================================



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
            self.x_coor += 0.1
        return x

    def generate(self):
        def descent(x): return 1.07 - math.tanh( x-1.5)
        with torch.no_grad():
            eps = descent(self.eps_coor)
            self.eps_coor += 3e-5
            noise = self.scale*self.disturbance()
            if self.ou_process:
                ou_bias = self.state
                noise -= self.scale*ou_bias
                self.state = ou_bias + noise
            else:
                self.state = noise
        return eps*self.max_action*self.state