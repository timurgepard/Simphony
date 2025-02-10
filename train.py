import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import random
import pickle
import time
import math
from symphony import Symphony, log_file



#==============================================================================================
#==============================================================================================
#===================================SCRIPT FOR TRAINING========================================
#==============================================================================================
#==============================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#global parameters
# environment type.
option = 2


explore_time = 5120
limit_step = 1000 #max steps per episode
limit_eval = 1000 #max steps per evaluation
num_episodes = 1000000
start_episode = 1 #number for the identification of the current episode
episode_rewards_all, episode_steps_all, test_rewards, Q_learning, average_steps = [], [], [], False, 0





if option == -1:
    env = gym.make('Pendulum-v1')
    env_test = gym.make('Pendulum-v1')

elif option == 0:
    env = gym.make('MountainCarContinuous-v0')
    env_test = gym.make('MountainCarContinuous-v0')

elif option == 1:
    env = gym.make('HalfCheetah-v4', render_mode="human")
    env_test = gym.make('HalfCheetah-v4')

elif option == 2:
    env = gym.make('Walker2d-v4', render_mode="human")
    env_test = gym.make('Walker2d-v4')

elif option == 3:
    env = gym.make('Humanoid-v4')
    env_test = gym.make('Humanoid-v4')

elif option == 4:
    limit_step = 300
    limit_eval = 300
    env = gym.make('HumanoidStandup-v4')
    env_test = gym.make('HumanoidStandup-v4')

elif option == 5:
    env = gym.make('Ant-v4', render_mode="human")
    env_test = gym.make('Ant-v4')


elif option == 6:
    env = gym.make('BipedalWalker-v3', render_mode="human")
    env_test = gym.make('BipedalWalker-v3')

elif option == 7:
    env = gym.make('BipedalWalkerHardcore-v3')
    env_test = gym.make('BipedalWalkerHardcore-v3')

elif option == 8:
    limit_step = 700
    limit_eval = 700
    env = gym.make('LunarLanderContinuous-v2')
    env_test = gym.make('LunarLanderContinuous-v2')

elif option == 9:
    limit_step = 300
    limit_eval = 200
    env = gym.make('Pusher-v4')
    env_test = gym.make('Pusher-v4')

elif option == 10:
    burst = True
    env = gym.make('Swimmer-v4')
    env_test = gym.make('Swimmer-v4')

elif option == 11:
    env = gym.make('Hopper-v4')
    env_test = gym.make('Hopper-v4')


state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

print('action space high', env.action_space.high)
max_action = torch.FloatTensor(env.action_space.high) if env.action_space.is_bounded() else 1.0

algo = Symphony(state_dim, action_dim, device, max_action)





#==============================================================================================
#==============================================================================================
#==========================================RECOVERY===========================================
#==============================================================================================
#==============================================================================================

def hard_recovery(algo, replay_buffer, size):
    algo.replay_buffer.states[:size] = replay_buffer.states[:size]
    algo.replay_buffer.actions[:size] = replay_buffer.actions[:size]
    algo.replay_buffer.rewards[:size] = replay_buffer.rewards[:size]
    algo.replay_buffer.next_states[:size] = replay_buffer.next_states[:size]
    algo.replay_buffer.dones[:size] = replay_buffer.dones[:size]
    algo.replay_buffer.length = len(replay_buffer.indices)




#==============================================================================================
#==============================================================================================
#==========================================TESTING=============================================
#==============================================================================================
#==============================================================================================

#testing model
def testing(env, limit_step, test_episodes, current_step=0, save_log=False):
    if test_episodes<1: return
    print("Validation... ", test_episodes, " epsodes")
    episode_return = []

    for test_episode in range(test_episodes):

        state = env.reset()[0]
        rewards = []

        for steps in range(1,limit_step+1):

            r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
            torch.manual_seed(r1)
            np.random.seed(r2)
            random.seed(r3)

            action = algo.select_action(state, mean=True)
            next_state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            state = next_state

            if done or truncated: break

        episode_return.append(np.sum(rewards))

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode+1}:, Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}")

    if save_log: log_file.write(str(current_step) + " : " + str(round(validate_return.item(), 2)) + "\n")


#==============================================================================================
#==============================================================================================
#=====================LOADING EXISTING MODELS, BUFFER and PARAMETERS===========================
#==============================================================================================
#==============================================================================================


total_steps = 0

try:
    print("loading buffer...")
    with open('data', 'rb') as file:
        dict = pickle.load(file)
        algo.replay_buffer = dict['buffer']
        #hard_recovery(algo, dict['buffer'], 200000+20000) # comment the previous line and chose a memory size to recover from old buffer
        algo.q_next_ema = dict['q_next_ema']
        episode_rewards_all = dict['episode_rewards_all']
        episode_steps_all = dict['episode_steps_all']
        total_steps = dict['total_steps']
        average_steps = dict['average_steps']
        if len(algo.replay_buffer)>=explore_time and not Q_learning: Q_learning = True
    print('buffer loaded, buffer length', len(algo.replay_buffer))

    start_episode = len(episode_steps_all)

except:
    print("problem during loading buffer")


try:
    print("loading models...")
    algo.nets.load_state_dict(torch.load('nets_model.pt', weights_only=True))
    algo.nets_target.load_state_dict(torch.load('nets_target_model.pt', weights_only=True))
    print('models loaded')
    #testing(env_test, limit_eval, 10)
except:
    print("problem during loading models")




#==============================================================================================
#==============================================================================================
#========================================EXPLORATION===========================================
#==============================================================================================
#==============================================================================================



if not Q_learning:
    log_file.clean()
    total_steps = 0

    while not Q_learning:
        rewards = []
        state = env_test.reset()[0]

        for steps in range(1, limit_step+1):
            total_steps += 1
        
            
            action = max_action.numpy()*np.random.uniform(-0.5, 0.75, size=action_dim)
            #action = algo.select_action(state)
            next_state, reward, done, truncated, info = env_test.step(action)
            rewards.append(reward)
            algo.replay_buffer.add(state, action, reward, next_state, done)
            if algo.replay_buffer.length>=explore_time and not Q_learning: Q_learning = True
            if Q_learning or done: break
            state = next_state
        Return = np.sum(rewards)
        print(f" Rtrn = {Return:.2f}")

    
    #total_steps = 0
    print("copying explore data, current length", algo.replay_buffer.length)
    algo.replay_buffer.fill()
    print("replay buffer length: ", algo.replay_buffer.length)



    


#==============================================================================================
#==============================================================================================
#=========================================TRAINING=============================================
#==============================================================================================
#==============================================================================================


print("started training")
#print(f"ReSine scale:\n {algo.actor.ffw[0].ffw[3].scale.cpu().detach().numpy()}")


for i in range(start_episode, num_episodes):

    rewards = []
    state = env.reset()[0]
    episode_steps = 0

    #----------------------------pre-processing------------------------------

    #--------------------2. CPU/GPU cooling ------------------
    time.sleep(0.75)


    for steps in range(1, limit_step+1):
        episode_steps += 1
        total_steps += 1

        # save models, data
        if (total_steps>=1250 and total_steps%1250==0):
            part = ""
            #part = "_"+str(total_steps/1000) if total_steps%300000==0 else ""
            testing(env_test, limit_step=limit_eval, test_episodes=50, current_step=total_steps, save_log=True)
            torch.save(algo.nets.state_dict(), 'nets_model'+ part +'.pt')
            torch.save(algo.nets_target.state_dict(), 'nets_target_model'+ part +'.pt')
            with open('data'+ part, 'wb') as file:
                pickle.dump({'buffer': algo.replay_buffer, 'q_next_ema': algo.q_next_ema, 'episode_rewards_all':episode_rewards_all, 'episode_steps_all':episode_steps_all, 'total_steps': total_steps, 'average_steps': average_steps}, file)
            

 
        action = algo.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        algo.replay_buffer.add(state, action, reward, next_state, done)
        algo.train()
        if done: break
        state = next_state

    episode_rewards_all.append(np.sum(rewards))
    average_reward = np.mean(episode_rewards_all[-100:])

    episode_steps_all.append(episode_steps)
    average_steps = np.mean(episode_steps_all[-100:])



    print(f"Ep {i}: Rtrn = {episode_rewards_all[-1]:.2f} | ep steps = {episode_steps} | total_steps = {total_steps}")

    log_file.write_opt(str(i) + " : " + str(round(episode_rewards_all[-1], 2)) + " : step : " + str(total_steps) + "\n")


