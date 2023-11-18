# Symphony


Sorry for solving Mujoco and other environments, I have group of orphans who are less protected at this period, I need to support them, and my family, of course

I wrote a short book with a careful explanation: https://www.amazon.com/dp/B0CKYWHPF5
email: timur.ishuov@gmail.com

if you want to support me: 4400 4301 8810 7871 (VISA)

The algorithm is cleaned, 248 lines, includes:

1. harmonization of the neural network
2. rectified Huber symmetrical/assymetrical error loss functions
3. "immediate" Advantage (but excessive training)
4. "movement is life" concept
5. careful TD3, element-wise minimum of 3 sub-nets
6. fading replay buffer: old transitions fade away gradually
7. without multi-agents, model-free (but can be added)

ver 2.0 includes:

8. reduced objective to learn Bellman's sum of dumped reward's variance
9. improve reward variance through immediate Advantage

   P.S.: My heart belongs to Jesus. Jesus is Love. Whoever seeks Him, finds Him...
   
   I want to say thanks to the "University of Szeged" for providing me space and desktop computer with AMD Ryzen 7 3800X and NVidia RTX 3060

![image](https://github.com/timurgepard/Simphony/assets/13238473/864a23b6-a2c8-4e83-b69c-497c4cd662c1)

| BipedalWalker-v3  | Humanoid-v4 (last update) |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/6c06b33b-5ea1-4443-8431-9bcf234e9167)  |  ![image](https://github.com/timurgepard/Simphony/assets/13238473/8684839b-bb1e-4b75-81f3-ad18751573cf) |

|  Ant-v4 | Walker-v4 |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/90cd49eb-e229-47db-998d-51c18b16850d)  | ![image](https://github.com/timurgepard/Simphony/assets/13238473/b9510d43-f8ab-462c-aa0e-6a398a7a2f8b)|







   
   additionally:
1. random initialization prevent the same initial states in the buffer
2. exploration-noise in the beginning

   
