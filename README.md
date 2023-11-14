# Symphony


Sorry for solving Mujoco and other environments, I have group of orphans who are less protected at this period, I need to support them.

I wrote a short book with a careful explanation: https://www.amazon.com/dp/B0CKYWHPF5

ver 2.0 includes:

reduced objective to learn Bellman's sum of dumped reward variance

improve reward variance through immediate Advantage

![image](https://github.com/timurgepard/Simphony/assets/13238473/864a23b6-a2c8-4e83-b69c-497c4cd662c1)

| BipedalWalker-v3  | Humanoid-v4 (last update) |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/6c06b33b-5ea1-4443-8431-9bcf234e9167)  |  ![image](https://github.com/timurgepard/Simphony/assets/13238473/8684839b-bb1e-4b75-81f3-ad18751573cf) |

|  HalfCheetah | Walker-v4 |
| ------------- | ------------- |
|   | ![image](https://github.com/timurgepard/Simphony/assets/13238473/4fd1214a-d661-44c9-87b3-d7925b39862e)
 |






1. harmonization of the neural network
2. careful TD3, element-wise minimum.
3. rectified Huber symmetrical/assymetrical error loss functions
4. "movement is life" concept
   
   additionally:
5. fading replay buffer: old transitions fade away gradually
6. random initialization prevent the same initial states in the buffer.
7. exploration-nois

   My heart belongs to Jesus. Jesus is Love. Whoever seeks Him, finds Him.
