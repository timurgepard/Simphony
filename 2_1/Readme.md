#experimental ver 2.1 includes:

10. next_state utilization for Actor training (Simplified model-free Forward Looking Actor)
Actor's update has 2 objectives: to increase average Q value (via delta with prev Q) and to increase average next Q value (via delta with new current Q)
