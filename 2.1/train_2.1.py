import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pickle
import time
from symphony_op_2_1 import Symphony, ReplayBuffer
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#global parameters
# environment type. Different Environments have some details that you need to bear in mind.
option = 1

explore_time = 5000
tr_between_ep_init = 15 # training between episodes
tr_per_step = 3 # training per frame
start_test = 250
limit_step = 2000 #max steps per episode
num_episodes = 10000000
start_episode = 0 #number for the identification of the current episode
total_rewards, total_steps, test_rewards, policy_training = [], [], [], False


hidden_dim = 256
max_action = 1.0
fade_factor = 7 # fading memory factor, 7 -remembers ~30% of the last transtions before gradual forgetting, 1 - linear forgetting, 10 - ~50% of transitions, 100 - ~70% of transitions.
stall_penalty = 0.07 # moving is life, stalling is dangerous, optimal value = 0.07, higher values can create extra vibrations.
capacity = "full" # short = 100k, medium=300k, full=500k replay buffer memory size.

#TD3 has one bottleneck, when it takes minimum between predictions, and we even use element-wise minimum between 3 sub-nets.
# each iteration it takes min -> min -> min, it can be compared with exp decaying function.
# it is well suited for tasks where high level of balancing is involved.
# but it is less suited for "overcoming tasks with low terminal rewards", if from 5 times agent scored -100.0, -100.0, -100.0, -100. and 30,
# minimum prediction will be value near -100.0 making an angent less interested to take any risky actions further.
# we take "anchored" average 0.7*min + 0.3*mean for "overcoming tasks" (BipedalWalkerHardcore), which unites advantages of TD3 and DDPG.
critics_average = False #takes "anchored" average (or average with min baseline) between Critic subnets, default minimum.
seq = 3


if option == 1:
    limit_step = 777
    critics_average = True
    fade_factor = 10
    capacity = "medium"
    env = gym.make('BipedalWalkerHardcore-v3')
    env_test = gym.make('BipedalWalkerHardcore-v3')


state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

print('action space high', env.action_space.high)
max_action = max_action*torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else max_action*1.0
replay_buffer = ReplayBuffer(state_dim, action_dim, capacity, device, seq, fade_factor, stall_penalty)
algo = Symphony(state_dim, action_dim, seq, hidden_dim, device, max_action, critics_average)



#used to create random initalization in Actor -> less dependendance on the specific random seed.
def init_weights(m):
    if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)


#testing model
def testing(env, limit_step, test_episodes):
    if test_episodes<1: return
    print("Validation... ", test_episodes, " epsodes")
    episode_return = []

    for test_episode in range(test_episodes):
        state = env.reset()[0]
        rewards_report = []
        states, actions, rewards, next_states, dones = [], [], [], [], []
        action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)

        for steps in range(0,seq):
            next_state, reward, done, info , _ = env.step(action)
            rewards_report.append(reward)

            states.append(state)
            actions.append(action)
            rewards.append([reward])
            next_states.append(next_state)
            dones.append([done])

            state = next_state

        for steps in range(1,limit_step+1):
            action = algo.select_action(states[-seq:])
            next_state, reward, done, info , _ = env.step(action)
            rewards_report.append(reward)

            states.append(state)
            actions.append(action)
            rewards.append([reward])
            next_states.append(next_state)
            dones.append([done])


            state = next_state
            if done: break

        episode_return.append(np.sum(rewards_report))

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

    start_episode = len(total_steps)

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




for i in range(start_episode, num_episodes):
    rewards_report = []
    state = env.reset()[0]

    states, actions, rewards, next_states, dones = [], [], [], [], []

    #----------------------------pre-processing------------------------------

    rb_len = len(replay_buffer)
    rb_len_treshold = 5000*tr_between_ep_init
    #--------------0. increase ep training: init + (1 to 100)-------------
    tr_between_ep = tr_between_ep_init
    if tr_between_ep_init>=100 and rb_len>=350000: tr_between_ep = rb_len//5000 # from 70 to 100
    if tr_between_ep_init<100 and rb_len>=rb_len_treshold: tr_between_ep = rb_len//5000
    #---------------------------1. processor releave --------------------------
    if policy_training: time.sleep(0.5)
     #---------------------2. decreases dependence on random seed: ---------------
    if not policy_training and rb_len<explore_time: algo.actor.apply(init_weights)
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)
    for steps in range(0, seq):
        next_state, reward, done, info, _ = env.step(action)
        rewards_report.append(reward)


        states.append(state)
        actions.append(action)
        rewards.append([reward])
        next_states.append(next_state)
        dones.append([done])

        state = next_state

    #------------------------------training------------------------------

    if policy_training: _ = [algo.train(replay_buffer.sample()) for x in range(tr_between_ep)]
        
    episode_steps = 0
    for steps in range(1, limit_step+1):
        episode_steps += 1

        if len(replay_buffer)>=explore_time and not policy_training:
            print("started training")
            policy_training = True
            _ = [algo.train(replay_buffer.sample()) for x in range(128)]

        #print(np.array().shape)
        action = algo.select_action(np.array(states[-seq:]))
        next_state, reward, done, info, _ = env.step(action)
        rewards_report.append(reward)


        states.append(state)
        actions.append(action)
        rewards.append([reward])
        next_states.append(next_state)
        dones.append([done])

        #print(rewards)
        
        replay_buffer.add(states[-seq:], actions[-seq:], rewards[-seq:], next_states[-seq:], dones[-seq:])
        if policy_training: _ = [algo.train(replay_buffer.sample()) for x in range(tr_per_step)]
        state = next_state
        if done: break


    total_rewards.append(np.sum(rewards_report))
    average_reward = np.mean(total_rewards[-100:])

    total_steps.append(episode_steps)
    average_steps = np.mean(total_steps[-100:])
            
    

    print(f"Ep {i}: Rtrn = {total_rewards[-1]:.2f} | ep steps = {episode_steps}")


    if policy_training:

        #--------------------saving-------------------------
        if (i%5==0): 
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            #print("saving... len = ", len(replay_buffer))
            with open('replay_buffer', 'wb') as file:
                pickle.dump({'buffer': replay_buffer, 'eps':algo.actor.eps, 'x_coor':algo.actor.x_coor, 'total_rewards':total_rewards, 'total_steps':total_steps}, file)
            #print(" > done")


        #-----------------validation-------------------------
        if (i>=start_test and i%50==0): testing(env_test, limit_step=1000, test_episodes=10)
              

#====================================================
# * Apart from the algo core, fade_factor, tr_between_ep and limit_steps are crucial parameters for speed of training.
#   E.g. limit_steps = 700 instead of 2000 introduce less variance and makes BipedalWalkerHardcore's Agent less discouraged to go forward.
#   high values in tr_between_ep can make a "stiff" agent, but sometimes it is helpful for straight posture from the beginning (Humanoid-v4).
