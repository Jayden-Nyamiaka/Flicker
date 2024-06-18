# Flicker: An Interactive Particle Simulation Game using CUDA OpenGL Interop
Authored by Jayden Nyamiaka

This covers the Description and Usage. For more information, reference
the PDF, and for implementation-specific details, reference the codebase.


## Description

This code runs a GPU-acceleratable particle simulation game that 
seamlessly integrates CUDA and OpenGL to get user-input, compute 
particle evolutions, and display results to the screen in real 
time. The behavior of particles is dependent on the user and needs
to be computed and rendered every frame, introducing significant 
challenges for continuously passing data between the CPU and GPU. 

The application currently supports 1 player and 3 different particle types.
Player: 
 - Controlled via WASD or Arrow Keys (simultaneously)
 - Color smoothly transitions over RGB spectrum
 - Can die on collision with particles. This should be on when using as a game.


Particles

Seeker (Particle 1):
This particle aims at the Player's position (at spawn time) and darts straight 
there with increasing acceleration. This particle changes colors from red to 
yellow depending on how fast it's traveling.

Cruiser (Particle 2):
This particle shoots in a simple (horizontal or vertical) direction but turns 
at seemingly random times. The turning is psuedo-random and frame-independent
implemented via a custom procedure that doesn't directly use any RNG. For 
implementation details, reference the code (and comments). This particle also 
changes colors depending on the direction its moving such that horizontally 
moving particles are green and vertically moving particles are violet.

Wanderer (Particle 3):
This particle wanders around by simulating Brownian motion, creating hot 
spots of dangerous unpredictability for the Player. Brownian motion models 
the random motion of particles suspended in a medium and is a mean zero, 
continuous process, often implemented using GPU acceleration due to its 
computational parallelism. This particle is also gray colored and slightly 
bigger than the other
For more information, refer to the code.



## Usage

The application doubles as both a game and a particle simulation, so the arguments we
opt to use for the application depend greatly on the use case. 

After building, the application can be called from the command line 
according to the following:
Usage: ./flicker.exe [options]
Options:
--help            Display this information.
--gpu-accel       Accelerate the simulation using the GPU. Otherwise, use the CPU.
                  Recommended for when using as a simulation.
--can-die         Has particle collisions kill the player & stop the simulation.
                  Otherwise, the player can't die. Recommended when using as a game.
--stagger         Stagger particle starts. Otherwise, all particles start simultaneously.
                  Recommended when using as a game.
--set-n n1 n2 n3  Manually set the number of each type of particle. Each n1 n2 n3 must
                  be a non-negative integer. The default is 32 32 32.
--preset [1-11]   Run the indicated simulation preset 1-11, discarding all other options.
                  The presets are as follows:
                    1:  Easy Game
                    2:  Hard Game
                    3:  Small Particle Simulation (All Particle Types)
                    4:  Large Particle Simulation (All Particle Types)
                    5:  Small Seeker Particle Simulation
                    6:  Large Seeker Particle Simulation
                    7:  Small Cruiser Particle Simulation
                    8:  Large Cruiser Particle Simulation
                    9:  Small Brownian Motion Simulation (Wanderer)
                    10: Large Brownian Motion Simulation (Wanderer)
                    11: Huge Seeker Particle Simulation


The presets list a few recommended calls depending on how you would like to use 
the application. Of course, you're encouraged to experiment with arguments and 
change the behavior of the application any way you like.

For all of the above presets, gpu_accel is toggled on depending on whether 
the application can benefit from GPU usage. In general, larger simulations 
benefitted from GPU acceleration due to the large number of particles that 
could all be evolved in parallel whereas the game and smaller simulations 
had too few particles to see any significant time improvements.