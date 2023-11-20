# Symphony


Sorry for solving Mujoco and other environments, I have group of orphans who are less protected at this period, I need to support them, and my family, of course

Acknowledgements: This algorithm was created in 3 years through my mom's and sister's financial support.

I wrote a short book with a careful explanation: https://www.amazon.com/dp/B0CKYWHPF5
email: timur.ishuov@gmail.com
if you want to support me: 4400 4301 8810 7871 (VISA)

The algorithm is cleaned, 265 lines, includes:

1. without multi-agents, model-free, off-policy (can work real-time) Actor and Critic
2. harmonics in neural networks
3. rectified Huber symmetrical/assymetrical error loss functions
4. "immediate" Advantage (but excessive training)
5. "movement is life" concept
6. careful TD3, element-wise minimum of 3 sub-nets
7. fading replay buffer: old transitions fade away gradually


ver 2.0 includes:

8. reduced objective to learn Bellman's sum of dumped reward's variance
9. improve reward variance through immediate Advantage

   P.S.: My heart belongs to Jesus. Jesus is Love. Whoever seeks Him, finds Him...
   
   I want to say thanks to the "University of Szeged" for providing me facilities to continue the research


![image](https://github.com/timurgepard/Simphony/assets/13238473/864a23b6-a2c8-4e83-b69c-497c4cd662c1)

All agents can be further improved if training continues, but only speed was concerned.

|  LunarLander-v2 | Animation |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/11cf2201-50e2-471b-849f-c609c794a7a7) |![LunarLander](https://github.com/timurgepard/Simphony/assets/13238473/351cae3a-95bf-46a1-be3a-f11506353444)

 |

|  Ant-v4 | Animation |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/90cd49eb-e229-47db-998d-51c18b16850d)  |![Ant](https://github.com/timurgepard/Simphony/assets/13238473/342f48a2-1610-43c4-86ad-8b3c3dc652b6)
 |

| Humanoid-v4 (ver 1.0)  | Animation |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/8684839b-bb1e-4b75-81f3-ad18751573cf) |  ![Humanoid](https://github.com/timurgepard/Simphony/assets/13238473/ba2117a0-e03c-4c33-aab8-2395ab42b118)
   |

| BipedalWalker-v3  | Animation |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/6c06b33b-5ea1-4443-8431-9bcf234e9167)  | |



|  Walker-v4 | Animation |
| ------------- | ------------- |
| ![image](https://github.com/timurgepard/Simphony/assets/13238473/b9510d43-f8ab-462c-aa0e-6a398a7a2f8b) |![Walker-2d](https://github.com/timurgepard/Simphony/assets/13238473/154abb7e-f0a5-4ce7-9813-466e01b3795f)
 |








   
   additionally:
1. slightly random initialization prevent the same initial states in the buffer
2. exploration-noise in the beginning

   
