import torch
import torch.nn as nn
import torch.optim as optim


def tril_init(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.tril(linear.weight))

# Zero out gradients
def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook

#Rectified Hubber Error Loss Function
def ReHE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Hubber Assymetric Error Loss Function
def ReHaE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class Input(nn.Module):
    def __init__(self, f_in, hidden_dim):
        super().__init__()

        self.nets = nn.ModuleList([nn.Linear(f_in, hidden_dim//4) for _ in range(4)])

        self.out = nn.Sequential(
            nn.Linear(f_in, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        xs = [net(x) for net in self.nets]
        return self.out(torch.cat(xs, dim=-1))



class Output(nn.Module):
    def __init__(self, hidden_dim, f_out):
        super().__init__()

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, f_out)
        )

    def forward(self, x):
        return self.out(x)



class Block(nn.Module):
    def __init__(self, time_intervals, vocab_embed, n_embed, tri_W):
        super().__init__()


        self.fft = nn.Sequential(
            nn.Linear(time_intervals, time_intervals, bias=None),
            nn.Linear(time_intervals, time_intervals, bias=None),
        )

        self.ffw1 = nn.Sequential(
            nn.Linear(n_embed, vocab_embed),
            Sin(),
            nn.Linear(vocab_embed, n_embed)
        )

        self.ffw2 = nn.Sequential(
            nn.Linear(n_embed, vocab_embed),
            Cos(),
            nn.Linear(vocab_embed, n_embed)
        )

        self.fft[0].apply(tril_init)
        self.fft[0].weight.register_hook(get_zero_grad_hook(tri_W))

        self.fft[1].apply(tril_init)
        self.fft[1].weight.register_hook(get_zero_grad_hook(tri_W))

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        B, T, E = x.shape
        x = self.ln1(x)
        x += self.fft(x.reshape(B, E, T)).reshape(B, T, E)
        x = self.ln2(x)
        return x + self.ffw1(x) + self.ffw2(x)
    


class DeepNeuralNetwork(nn.Module):
    def __init__(self, f_in, hidden_dim, f_out, device):
        super().__init__()

        self.input = Input(f_in, hidden_dim)
        tri = torch.tril(torch.ones((7, 7), dtype=torch.float32)).to(device)
        tri_W = tri/tri.sum(dim=1, keepdim=True)
        self.blocks = nn.Sequential(*[Block(7, hidden_dim, tri_W.detach()) for _ in range(7)])
        self.out = Output(hidden_dim, f_out)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.out(x)


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
            xs = [net(x)[:,-1,:].cpu().data.numpy().flatten() for net in self.nets]
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
    



