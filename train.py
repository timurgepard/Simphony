import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle
import time
from symphony import Symphony, ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#global parameters
#1 BipedalWalker, 2 Humanoid
option = 2

explore_time = 5000
tr_between_ep = 30 # training between episodes
tr_per_step = 3 # training per frame
start_test = 250
limit_step = 1000000 #max steps per episode
num_episodes = 1000000
start_episode = 0 #number for the identification of the current episode
total_rewards, total_steps, test_rewards, policy_training = [], [], [], False


hidden_dim = 256



if option == 1:
    env = gym.make('BipedalWalker-v3')
    env_test = gym.make('BipedalWalker-v3', render_mode="human")
    limit_step = 10000
elif option == 2:
    env = gym.make('Humanoid-v4')
    env_test = gym.make('Humanoid-v4', render_mode="human")




state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

print('action space high', env.action_space.high)

max_action = torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else 1.0
replay_buffer = ReplayBuffer(state_dim, action_dim, device)
algo = Symphony(state_dim, action_dim, hidden_dim, device, max_action)






#testing model
def testing(env, limit_step, test_episodes):
    if test_episodes<1: return
    print("Validation... ", test_episodes, " epsodes")
    episode_return = []

    for test_episode in range(test_episodes):
        state = env.reset()[0]
        rewards = []

        for steps in range(1,limit_step+1):
            action = algo.select_action(state)
            next_state, reward, done, info , _ = env.step(action)
            rewards.append(reward)
            state = next_state
            
            if done: break

        episode_return.append(np.sum(rewards))

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode}:, Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}")





#--------------------loading existing models, replay_buffer, parameters-------------------------

try:
    print("loading buffer...")
    with open('replay_buffer', 'rb') as file:
        dict = pickle.load(file)
        algo.actor.eps = dict['eps']
        algo.actor.x_coor = dict['x_coor']
        replay_buffer = dict['buffer']
        total_rewards = dict['total_rewards']
        total_steps = dict['total_steps']
        if len(replay_buffer)>=explore_time and not policy_training: policy_training = True
    print('buffer loaded, buffer length', len(replay_buffer))

    start_episode = len(total_steps) - 1

except:
    print("problem during loading buffer")

try:
    print("loading models...")
    algo.actor.load_state_dict(torch.load('actor_model.pt'))
    algo.critic.load_state_dict(torch.load('critic_model.pt'))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt'))
    print('models loaded')
    testing(env_test, limit_step, 10)
except:
    print("problem during loading models")

#-------------------------------------------------------------------------------------


#used to create random initalization in Actor -> less dependendance on the specific random seed.
def init_weights(m):
    if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)

for i in range(start_episode, num_episodes):
    rewards = []
    state = env.reset()[0]

    

    #---------------------------1. processor releave --------------------------
    if policy_training: time.sleep(0.5)
     #---------------------2. decreases dependence on random seed: ---------------
    if not policy_training and len(replay_buffer)<explore_time: algo.actor.apply(init_weights)
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)
    for steps in range(0, 4):
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        state = next_state

    #------------------------------training------------------------------

    if policy_training: _ = [algo.train(replay_buffer.sample()) for x in range(tr_between_ep)]
        

    for steps in range(1, limit_step+1):

        if len(replay_buffer)>=explore_time and not policy_training:
            print("started training")
            policy_training = True
            _ = [algo.train(replay_buffer.sample()) for x in range(128)]

        action = algo.select_action(state)
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        replay_buffer.add(state, action, reward, next_state, done)
        if policy_training: _ = [algo.train(replay_buffer.sample()) for x in range(tr_per_step)]
        state = next_state
        if done: break
            

    total_rewards.append(np.sum(rewards))
    average_reward = np.mean(total_rewards[-100:])

    episode_steps = steps
    total_steps.append(episode_steps)
    average_steps = np.mean(total_steps[-100:])
            
    

    print(f"Ep {i}: Rtrn = {total_rewards[i]:.2f} | ep steps = {episode_steps}")


    if policy_training:

        #--------------------saving-------------------------
        if (i%5==0): 
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            #print("saving... len = ", len(replay_buffer), end="")
            with open('replay_buffer', 'wb') as file:
                pickle.dump({'buffer': replay_buffer, 'eps':algo.actor.eps, 'x_coor':algo.actor.x_coor, 'total_rewards':total_rewards, 'total_steps':total_steps}, file)
            #print(" > done")


        #-----------------validation-------------------------
        if (i>=start_test and i%50==0): testing(env_test, limit_step=2000, test_episodes=10)
              
        #====================================================
