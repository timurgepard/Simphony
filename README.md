# Symphony


![image](https://github.com/timurgepard/Simphony/assets/13238473/864a23b6-a2c8-4e83-b69c-497c4cd662c1)

BipedalWalker-v3 in robot_like mode:

![Screenshot_50_ep](https://github.com/timurgepard/Simphony/assets/13238473/5f677487-18d3-4bcf-b41e-4d1f4745b724)

I wrote a short book with a careful explanation: https://www.amazon.com/dp/B0CKYWHPF5

1. harmonization of the neural network
2. careful TD3, element-wise minimum of 3 predictions.
3. rectified Huber symmetrical/assymetrical error loss functions
   
   additionally:
4. fading replay buffer: old transitions fade away gradually
5. emphasis on the beginning of training, number of steps grows exponentially
6. random initialization prevent the same initial states in the buffer.
7. exploration-noise, reward/punishment, etc

   My heart belongs to Jesus. Jesus is Love. Whoever seeks Him, finds Him.
