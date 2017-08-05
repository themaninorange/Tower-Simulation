# Tower-Simulation

This is not stats research.  I wanted to use a genetic algorithm to build up steel structures.  This works in two parts.

1. The structure is simulated as a collection of point particles using CUDA C.
2. Metrics from each simulated is recorded, and Python scripts manage the population of towers, preserving the towers that reach the highest while having the fewest parts of the tower be compromised.

Success | Failure
:-------------------------:|:-------------------------:
![Wobbly, but okay](https://github.com/themaninorange/Tower-Simulation/blob/master/visuals/hextower.gif "Alright")  | ![Too tall](https://github.com/themaninorange/Tower-Simulation/blob/master/visuals/hextower2.gif "Bad")

## Simulation Physics

The physics are handled by an NVIDIA graphics card.  Each node is passed to a thread in the kernel call.  Then, I use Euler's method to alternately update the positions, velocities, and forces.  I also keep track of whether the beam is currently under too much stress.  After each update, the positions can be sent back to the CPU to make a frame corresponding to that moment.  Failing beams are colored red.

Once the maximum velocity of any node drops below a threshold, the tower is considered stable.  At this point, all of the positions and connection informations are recorded into a file that can be read again to be part of another simulation.

## Populations

The Python scripts initialize a population of a given size with random positions, connections, and constants.  These towers are usually very bad.  After I have a population of stable towers, I iterate through generations, reading the best half of the towers, splicing them together, and mutating them.  I delete the worst half, and repopulate.  This can be done as many times as we want.
