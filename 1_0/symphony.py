import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import random
import torch.nn.functional as F
import torch.jit as jit
import os, re


#==============================================================================================
#==============================================================================================
#=========================================LOGGING=============================================
#==============================================================================================
#==============================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to continue writing to the same history file and derive its name. This function created with the help of ChatGPT
def extract_r1_r2_r3():
    pattern = r'history_(\d+)_(\d+)_(\d+)\.log'

    # Iterate through the files in the given directory
    for filename in os.listdir():
        # Match the filename with the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the numbers r1, r2, and r3 from the filename
            return map(int, match.groups())
    return None


#write or append to the history log file
class LogFile(object):
    def __init__(self, log_name_main, log_name_opt):
        self.log_name_main = log_name_main
        self.log_name_opt = log_name_opt
    def write(self, text):
        with open(self.log_name_main, 'a+') as file:
            file.write(text)
    def write_opt(self, text):
        with open(self.log_name_opt, 'a+') as file:
            file.write(text)
    def clean(self):
        with open(self.log_name_main, 'w') as file:
            file.write("")
        with open(self.log_name_opt, 'w') as file:
            file.write("")


numbers = extract_r1_r2_r3()
if numbers != None:
    # derive random numbers from history file
    r1, r2, r3 = numbers
else:
    # generate new random seeds
    r1, r2, r3 = random.randint(0,10), random.randint(0,10), random.randint(0,10)

torch.manual_seed(r1)
np.random.seed(r2)
random.seed(r3)

print(r1, ", ", r2, ", ", r3)

log_name_main = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_name_opt = "episodes_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_file = LogFile(log_name_main, log_name_opt)




#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================




#Rectified Huber Symmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHSE(jit.ScriptModule):
    def __init__(self):
        super(ReHSE, self).__init__()

    @jit.script_method
    def forward(self, e, k:float):
        ae = torch.abs(e) + 1e-6
        ae = ae**k*torch.tanh(k*ae/2)
        return ae.mean()


#Rectified Huber Asymmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHAE(jit.ScriptModule):
    def __init__(self):
        super(ReHAE, self).__init__()

    @jit.script_method
    def forward(self, e, k:float):
        e = e + 1e-6
        e = torch.abs(e)**k*torch.tanh(k*e/2)
        return e.mean()



#Linear Layer followed by Silent Dropout
# nn.Module -> JIT C++ graph
class LinearSDropout(jit.ScriptModule):
    def __init__(self, f_in, f_out, p=0.5):
        super(LinearSDropout, self).__init__()
        self.ffw = nn.Linear(f_in, f_out)
        self.p = p

    @jit.script_method
    def forward(self, x):
        x = self.ffw(x)
        #Silent Dropout function created with the help of ChatGPT
        # It is not recommended to use JIT compilation decorator with online random generator as Symphony updates seeds each time
        # We did exception only for this module as it is used inside neural networks.
        mask = (torch.rand_like(x) > self.p).float()
        return  mask * x + (1.0-mask) * x.detach()



#ReSine Activation Function
# nn.Module -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()
        self.s = nn.Parameter(data=2.0*torch.rand(hidden_dim)-1.0, requires_grad=True)

    @jit.script_method
    def forward(self, x):
        s = torch.sigmoid(self.s)
        x = s*torch.sin(x/s)
        return x/(1+torch.exp(-1.5*x/s))





#Shared Feed Forward Module
# nn.Module -> JIT C++ graph
class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, f_out, p_out=0.5):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 384),
            ReSine(384),
            LinearSDropout(384, 256, 0.5),
            LinearSDropout(256, f_out, 0.5)
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)



# nn.Module -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()



        self.a = FeedForward(state_dim, action_dim)
        self.a_max = nn.Parameter(data= max_action, requires_grad=False)


        self.qA = FeedForward(state_dim+action_dim, 128)
        self.qB = FeedForward(state_dim+action_dim, 128)
        self.qC = FeedForward(state_dim+action_dim, 128)

        self.qnets = nn.ModuleList([self.qA, self.qB, self.qC])

        self.max_action = max_action
        self.action_dim = action_dim




    #========= Actor Forward Pass =========
    @jit.script_method
    def actor(self, state):
        return self.a_max*torch.tanh(self.a(state)/self.a_max)



    #========= Critic Forward Pass =========
    # take 3 distributions and concatenate them
    @jit.script_method
    def critic(self, state, action):
        x = torch.cat([state, action], -1)
        return torch.cat([qnet(x) for qnet in self.qnets], dim=-1)


    # take average in between min and mean
    @jit.script_method
    def critic_soft(self, state, action):
        x = self.critic(state, action).mean(dim=-1, keepdim=True)
        return x, x.detach()
        



# Define the algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, device, max_action=1.0):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device)

        self.nets = ActorCritic(state_dim, action_dim, max_action=max_action).to(device)
        self.nets_target = ActorCritic(state_dim, action_dim, max_action=max_action).to(device)
        self.nets_target.load_state_dict(self.nets.state_dict())

        self.learning_rate = 3e-4

        self.nets_optimizer = optim.RMSprop(self.nets.parameters(), lr=self.learning_rate)

        self.rehse = ReHSE()
        self.rehae = ReHAE()


        self.max_action = max_action
      

        self.tau = 0.005
        self.tau_ = 1.0 - self.tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device


        self.q_next_ema = 0.0



    def select_action(self, state, mean=False):
        state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
        with torch.no_grad(): action = self.nets.actor(state)
        return action.cpu().data.numpy().flatten()


    def train(self):
        # decreases dependence on random seeds:
        r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
        torch.manual_seed(r1)
        np.random.seed(r2)
        random.seed(r3)

        for _ in range(5): self.update()



    def update(self):

        state, action, reward, next_state, done = self.replay_buffer.sample()
        self.nets_optimizer.zero_grad(set_to_none=True)
        k = 1/5


        with torch.no_grad():
            for target_param, param in zip(self.nets_target.qnets.parameters(), self.nets.qnets.parameters()):
                target_param.data.copy_(self.tau_*target_param.data + self.tau*param.data)

        next_action = self.nets.actor(next_state)
        q_next_target, q_next_target_value = self.nets_target.critic_soft(next_state, next_action)
        q = 0.01 * reward + (1-done) * 0.99 * q_next_target_value
        qs = self.nets.critic(state, action)


        q_next_ema = self.tau_ * self.q_next_ema + self.tau * q_next_target_value
        nets_loss = -self.rehae(q_next_target-q_next_ema, k) + self.rehse(q-qs, k)

        nets_loss.backward()
        self.nets_optimizer.step()
        self.q_next_ema = q_next_ema.mean()




       



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device):

        #Normalized index conversion into fading probabilities
        def fade(norm_index):
            weights = np.tanh(70*norm_index**30 + 0.02*norm_index**2) # linear / -> non-linear _/‾
            return weights/np.sum(weights) #probabilities


        self.capacity, self.length, self.device = 384000, 0, device
        self.batch_size = 256
        self.random = np.random.default_rng()
        self.indexes = np.arange(0, self.capacity, 1)
        self.probs = fade(self.indexes/self.capacity)

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)


    def fill(self):
        def repeat(tensor, times):
            temp = tensor[:self.length]
            return temp.repeat(times, 1)

        times = self.capacity//self.length

        self.states = repeat(self.states, times)
        self.actions = repeat(self.actions, times)
        self.rewards = repeat(self.rewards, times)
        self.next_states = repeat(self.next_states, times)
        self.dones = repeat(self.dones, times)

        self.length = times*self.length



    def add(self, state, action, reward, next_state, done):
        if done: self.dones[-2:] = torch.tensor([done], dtype=torch.float32, device=self.device)
        repeat = 5 if done else 1
        for _ in range(repeat):
            if self.length<self.capacity: self.length += 1

            idx = self.length-1
            
            self.states[idx,:] = torch.tensor(state, dtype=torch.float32, device=self.device)
            self.actions[idx,:] = torch.tensor(action, dtype=torch.float32, device=self.device)
            self.rewards[idx,:] = torch.tensor([reward], dtype=torch.float32, device=self.device)
            self.next_states[idx,:] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            self.dones[idx,:] = torch.tensor([done], dtype=torch.float32, device=self.device)


            if self.length>=self.capacity:
                self.states = torch.roll(self.states, shifts=-1, dims=0)
                self.actions = torch.roll(self.actions, shifts=-1, dims=0)
                self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
                self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
                self.dones = torch.roll(self.dones, shifts=-1, dims=0)



   
    # Do not use any decorators with random generators (Symphony updates seed each time)
    def sample(self):
        indices = self.random.choice(self.indexes, p=self.probs, size=self.batch_size)


        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


    def __len__(self):
        return self.length
