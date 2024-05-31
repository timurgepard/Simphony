import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import random
import pickle
import time
from symphony import Symphony, log_file


#==============================================================================================
#==============================================================================================
#=========================================TRAINING=============================================
#==============================================================================================
#==============================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#global parameters
# environment type. Different Environments have some details that you need to bear in mind.
option = 3





explore_time = 1000
tr_per_step = 3 # actor's updates per frame/step
limit_step = 384 #max steps per episode
limit_eval = 1000 #max steps per evaluation
num_episodes = 1000000
start_episode = 0 #number for the identification of the current episode
episode_rewards_all, episode_steps_all, test_rewards, Q_learning = [], [], [], False


hidden_dim = 384
capacity = 96000
fade_factor = 10 # fading memory factor, 7 -remembers ~30% of the last transtions before gradual forgetting, 1 - linear forgetting, 10 - ~50% of transitions, 100 - ~70% of transitions.
tau = 0.005
lambda_r = 0.01 # base alpha for moving is life




if option == -1:
    env = gym.make('Pendulum-v1')
    env_test = gym.make('Pendulum-v1')

elif option == 0:
    env = gym.make('MountainCarContinuous-v0')
    env_test = gym.make('MountainCarContinuous-v0')

elif option == 1:
    env = gym.make('HalfCheetah-v4')
    env_test = gym.make('HalfCheetah-v4', render_mode="human")

elif option == 2:
    env = gym.make('Walker2d-v4')
    env_test = gym.make('Walker2d-v4')

elif option == 3:
    env = gym.make('Humanoid-v4')
    env_test = gym.make('Humanoid-v4')

elif option == 4:
    limit_step = 300
    limit_eval = 300
    env = gym.make('HumanoidStandup-v4')
    env_test = gym.make('HumanoidStandup-v4', render_mode="human")

elif option == 5:
    env = gym.make('Ant-v4')
    env_test = gym.make('Ant-v4', render_mode="human")
    #Ant environment has problem when Ant is flipped upside down and it is not detected (rotation around x is not checked, only z coordinate), we can check to save some time:
    angle_limit = 0.4
    #less aggressive movements -> faster learning but less final speed

elif option == 6:
    env = gym.make('BipedalWalker-v3')
    env_test = gym.make('BipedalWalker-v3')

elif option == 7:
    env = gym.make('BipedalWalkerHardcore-v3')
    env_test = gym.make('BipedalWalkerHardcore-v3')

elif option == 8:
    limit_step = 700
    limit_eval = 700
    env = gym.make('LunarLanderContinuous-v2')
    env_test = gym.make('LunarLanderContinuous-v2', render_mode="human")

elif option == 9:
    limit_step = 300
    limit_eval = 200
    env = gym.make('Pusher-v4')
    env_test = gym.make('Pusher-v4', render_mode="human")

elif option == 10:
    burst = True
    env = gym.make('Swimmer-v4')
    env_test = gym.make('Swimmer-v4', render_mode="human")



state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]

print('action space high', env.action_space.high)
max_action = torch.FloatTensor(env.action_space.high).to(device) if env.action_space.is_bounded() else 1.0

algo = Symphony(state_dim, action_dim, hidden_dim, device, max_action, tau, capacity, fade_factor, lambda_r, explore_time)



#used to create random initalization in Actor -> less dependendance on the specific random seed.
def init_weights(m):
    if isinstance(m, nn.Linear): torch.nn.init.xavier_uniform_(m.weight)



#testing model
def testing(env, limit_step, test_episodes, current_step=0, save_log=False):
    if test_episodes<1: return
    print("Validation... ", test_episodes, " epsodes")
    episode_return = []

    for test_episode in range(test_episodes):

        #---------------------1. decreases dependence on random seed: ---------------
        r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
        #print(r1, ", ", r2, ", ", r3)
        torch.manual_seed(r1)
        np.random.seed(r2)
        random.seed(r3)

        state = env.reset()[0]
        rewards = []

        for steps in range(1,limit_step+1):
            action = algo.select_action(state, mean=True)
            next_state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            state = next_state

            if done or truncated: break

        episode_return.append(np.sum(rewards))

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode+1}:, Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}")

    if save_log: log_file.write(str(current_step) + ": " + str(round(validate_return.item(), 2)) + "\n")




#--------------------loading existing models, buffer, parameters-------------------------
total_steps = 0

try:
    print("loading buffer...")
    with open('data', 'rb') as file:
        dict = pickle.load(file)
        algo.replay_buffer = dict['buffer']
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
    algo.actor.load_state_dict(torch.load('actor_model.pt'))
    algo.critic.load_state_dict(torch.load('critic_model.pt'))
    algo.critic_target.load_state_dict(torch.load('critic_target_model.pt'))
    print('models loaded')
    #testing(env_test, limit_eval, 10)
except:
    print("problem during loading models")

#-------------------------------------------------------------------------------------
log_file.write("experiment_started\n")

for i in range(start_episode, num_episodes):

    rewards = []
    state = env.reset()[0]
    episode_steps = 0

    #----------------------------pre-processing------------------------------
    #---------------------1. decreases dependence on random seed: ---------------
    r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
    #print(r1, ", ", r2, ", ", r3)
    torch.manual_seed(r1)
    np.random.seed(r2)
    random.seed(r3)
    #-----------------2. initialize Actor with random weights during exploration------------
    if not Q_learning and total_steps<explore_time: algo.actor.apply(init_weights)
    #--------------------2. CPU/GPU cooling ------------------
    if Q_learning: time.sleep(0.5)
    #-----------3. slighlty random initial configuration as in OpenAI Pendulum----
    """
    action = 0.3*max_action.to('cpu').numpy()*np.random.uniform(-1.0, 1.0, size=action_dim)
    for steps in range(0, 2):
        next_state, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        state = next_state
    """
    
        



    for steps in range(1, limit_step+1):
        episode_steps += 1
        total_steps += 1

        if (total_steps>=2500 and total_steps%2500==0): testing(env_test, limit_step=limit_eval, test_episodes=25, current_step=total_steps, save_log=True)

        if total_steps>=1000 and not Q_learning:
            print("started training")
            Q_learning = True

 

        action = algo.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        algo.replay_buffer.add(state, action, reward, next_state, done)
        if Q_learning: algo.train(tr_per_step+algo.replay_buffer.length//25600)
        state = next_state
        if done or (not Q_learning and steps>=100): break


    episode_rewards_all.append(np.sum(rewards))
    average_reward = np.mean(episode_rewards_all[-50:])

    episode_steps_all.append(episode_steps)
    average_steps = np.mean(episode_steps_all[-50:])

    #if Q_learning: limit_step = int(average_steps + average_steps/3)


    print(f"Ep {i}: Rtrn = {episode_rewards_all[-1]:.2f} | ep steps = {episode_steps} | total_steps = {total_steps} | alpha = {1}")#{round(algo.actor.ffw[0].ffw[2].var_scale.item(), 3)}")


    if Q_learning:
        
        #--------------------saving-------------------------
        if (i%5==0): 
            torch.save(algo.actor.state_dict(), 'actor_model.pt')
            torch.save(algo.critic.state_dict(), 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), 'critic_target_model.pt')
            #print("saving... len = ", len(algo.replay_buffer))
            with open('data', 'wb') as file:
                pickle.dump({'buffer': algo.replay_buffer, 'episode_rewards_all':episode_rewards_all, 'episode_steps_all':episode_steps_all, 'total_steps': total_steps, 'average_steps': average_steps}, file)
            #print(" > done")
        
