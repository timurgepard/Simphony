# experimental ver 2.1 includes:

Next State utilization for Actor training (Simplified model-free Forward Looking Actor). Actor's update has 2 objectives: 

1. to increase average Q value (via delta with prev Q)
2. to increase average next Q value (via delta with new current Q)
