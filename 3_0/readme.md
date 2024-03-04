# Symphony 3.0 Robotics

Top scores (not average, under development)
| BipedalWalker-v3  | Animation |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/8abc0713-d6f2-4a8d-930e-f7158aa631aa)  | ![BipedalWalker](https://github.com/timurgepard/Simphony/assets/13238473/725371e6-9f41-43dc-8ae6-188ad01642ac)|

# Symphony 3.0 Robotics + Simplified Dreamer

under development :
Symphony is off-policy algorithm with Replay Buffer.

1) First motivation was - to increase sample efficiency we can do Dreaming in between episodes. after exploration/pre-training, and after each episode:
   to do roll-outs based not by Environment step, but Dreamer step. We send to FeedForwardTransformer (see below) 7 first steps from the last episode cache to predict next state.
   We do 1000 roll-outs of 7 init steps and 40 predicted steps (47 in total), and do On-policy training using only this data (without Replay Buffer)

   key obstacle: After Critic-Actor's update if we return back to training using Replay Buffer, weights will shift towards old data from the Replay Buffer.
   Alternatively, we can populate Replay Buffer with this data, but it is coming from trained Environment Model, and can be not 100% precise. Our Transformer Model should be very accurate.

   Accuracy is the hardest challenge to solve, as number of transitions can be not sufficient enough. 

(I am giving 10$ RL lessons on this algorithm and in general, if you want to participate timur.ishuov@gmail.com)
[<img src="https://github.com/timurgepard/Simphony/assets/13238473/7fcb2907-0741-4aa9-9b7c-5da7b25bc330">](https://www.youtube.com/watch?v=_lIypdb3eHs)





FeedForward Only Transformer:


[<img src="https://github.com/timurgepard/Simphony/assets/13238473/849ec01d-13b4-4fb6-9efd-b6bcb97bb553">](https://www.youtube.com/watch?v=7VNAL4YQEqs)


