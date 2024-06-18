/* Flicker: An Interactive Particle Simulation Game using CUDA OpenGL Interop
 * 
 * Authored by Jayden Nyamiaka
 * 
 * Used Nvidia Sample's Simple GL as Base Structure 
 * 
 * 
 * Description
 * /////////////////////////////////////////////////////////////////////////////
 * This code runs a GPU-acceleratable particle simulation game that 
 * seamlessly integrates CUDA and OpenGL to get user-input, compute 
 * particle evolutions, and display results to the screen in real 
 * time. The behavior of particles is dependent on the user and needs
 * to be computed and rendered every frame, introducing significant 
 * challenges for continuously passing data between the CPU and GPU. 
 * 
 * The application currently supports 1 player and 3 different particle types.
 * Player: 
 *  - Controlled via WASD or Arrow Keys (simultaneously)
 *  - Color smoothly transitions over RGB spectrum
 *  - Can die on collision with particles. This should be on when using as a game.
 * 
 * 
 * Particles
 * Seeker (Particle 1):
 * This particle aims at the Player's position (at spawn time) and darts straight 
 * there with increasing acceleration. This particle changes colors from red to 
 * yellow depending on how fast it's traveling.
 * 
 * Cruiser (Particle 2):
 * This particle shoots in a simple (horizontal or vertical) direction but turns 
 * at seemingly random times. The turning is psuedo-random and frame-independent
 * implemented via a custom procedure that doesn't directly use any RNG. For 
 * implementation details, reference the code (and comments). This particle also 
 * changes colors depending on the direction its moving such that horizontally 
 * moving particles are green and vertically moving particles are violet.
 * 
 * Wanderer (Particle 3):
 * This particle wanders around by simulating Brownian motion, creating hot 
 * spots of dangerous unpredictability for the Player. Brownian motion models 
 * the random motion of particles suspended in a medium and is a mean zero, 
 * continuous process, often implemented using GPU acceleration due to its 
 * computational parallelism. This particle is also gray colored and slightly 
 * bigger than the others.

 *
 * For more information, refer to the code.
 * 
 * 
 * Usage
 * /////////////////////////////////////////////////////////////////////////////
 * The application doubles as both a game and a particle simulation, so the arguments we
 * opt to use for the application depend greatly on the use case. 
 * 
 * After building, the application can be called from the command line 
 * according to the following:
 * Usage: ./flicker.exe [options]
 * Options:
 * --help            Display this information.
 * --gpu-accel       Accelerate the simulation using the GPU. Otherwise, use the CPU.
 *                   Recommended for when using as a simulation.
 * --can-die         Has particle collisions kill the player & stop the simulation.
 *                   Otherwise, the player can't die. Recommended when using as a game.
 * --stagger         Stagger particle starts. Otherwise, all particles start simultaneously.
 *                   Recommended when using as a game.
 * --set-n n1 n2 n3  Manually set the number of each type of particle. Each n1 n2 n3 must
 *                   be a non-negative integer. The default is 32 32 32.
 * --preset [1-11]   Run the indicated simulation preset 1-11, discarding all other options.
 *                   The presets are as follows:
 *                     1:  Easy Game
 *                     2:  Hard Game
 *                     3:  Small Particle Simulation (All Particle Types)
 *                     4:  Large Particle Simulation (All Particle Types)
 *                     5:  Small Seeker Particle Simulation
 *                     6:  Large Seeker Particle Simulation
 *                     7:  Small Cruiser Particle Simulation
 *                     8:  Large Cruiser Particle Simulation
 *                     9:  Small Brownian Motion Simulation (Wanderer)
 *                     10: Large Brownian Motion Simulation (Wanderer)
 *                     11: Huge Seeker Particle Simulation
 * 
 * 
 * The presets list a few recommended calls depending on how you would like to use 
 * the application. Of course, you're encouraged to experiment with arguments and 
 * change the behavior of the application any way you like.
 * 
 * For all of the above presets, gpu_accel is toggled on depending on whether 
 * the application can benefit from GPU usage. In general, larger simulations 
 * benefitted from GPU acceleration due to the large number of particles that 
 * could all be evolved in parallel whereas the game and smaller simulations 
 * had too few particles to see any significant time improvements.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <math.h>
#include <random>

#define _USE_MATH_DEFINES


#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>      // includes cuRand, cuda random support

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

#define SQRT2DIV2 0.70710678118f // Used as diagonal speed factor

#define MAX(a,b) ((a > b) ? a : b)


// Toggle GPU Acceleration On and Off (Can also be changed from Cmd Line)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! Command Line Arguments (Currently set to default)
////////////////////////////////////////////////////////////////////////////////

// TODO Fix computeFPS
// TODO: R0Un Test to show time performance

// Count for each type of particle and total
// Defaults to the following values if not specified on cmd line
bool gpu_accel = false;
bool player_can_die = false;
bool particle_stagger_start = false;
unsigned int n_seekers =    32;     // Particle 1
unsigned int n_cruisers =   32;     // Particle 2
unsigned int n_wanderers =  32;     // Particle 3 1000000
unsigned int n_particles = n_seekers + n_cruisers + n_wanderers;


////////////////////////////////////////////////////////////////////////////////
//! Command Line Argument Functions
////////////////////////////////////////////////////////////////////////////////
void printUsage(char *exeName) {
    printf("Usage: %s [options]\n", exeName);
    printf("Options:\n");
    printf("--help            Display this information.\n");
    printf("--gpu-accel       Accelerate the simulation using the GPU. Otherwise, use the CPU. \n");
    printf("                  Recommended for when using as a simulation.\n");
    printf("--can-die         Has particle collisions kill the player & stop the simulation. \n");
    printf("                  Otherwise, the player can't die. Recommended when using as a game.\n");
    printf("--stagger         Stagger particle starts. Otherwise, all particles start simultaneously.\n");
    printf("                  Recommended when using as a game.\n");
    printf("--set-n n1 n2 n3  Manually set the number of each type of particle. Each n1 n2 n3 must \n");
    printf("                  be a non-negative integer. The default is %u %u %u.\n", n_seekers, n_cruisers, n_wanderers);
    printf("--preset [1-11]   Run the indicated simulation preset 1-11, discarding all other options. \n");
    printf("                  The presets are as follows: \n");
    printf("                    1:  Easy Game \n");
    printf("                    2:  Hard Game \n");
    printf("                    3:  Small All Particle Simulation \n");
    printf("                    4:  Large All Particle Simulation \n");
    printf("                    5:  Small Seeker Particle Simulation \n");
    printf("                    6:  Large Seeker Particle Simulation \n");
    printf("                    7:  Small Cruiser Particle Simulation \n");
    printf("                    8:  Large Cruiser Particle Simulation \n");
    printf("                    9:  Small Brownian Motion Simulation (Wanderer) \n");
    printf("                    10: Large Brownian Motion Simulation (Wanderer) \n");
    printf("                    11: Huge Seeker Particle Simulation \n");
}
void setSimulationConfig(bool gpu, bool can_die, bool stagger, int ns, int nc, int nw) {
    gpu_accel = gpu;
    player_can_die = can_die;
    particle_stagger_start = stagger;
    n_seekers =    ns;
    n_cruisers =   nc;
    n_wanderers =  nw;
    n_particles = n_seekers + n_cruisers + n_wanderers;
}
void setSimulationPreset(int i, char *execName) {
    switch (i) {
        case 1: // Easy Game
            setSimulationConfig(false, true, true, 16, 16, 16);
            break;
        case 2: // Hard Game
            setSimulationConfig(false, true, true, 32, 32, 32);
            break;
        case 3: // Small All Particle Simulation
            setSimulationConfig(false, false, false, 256, 256, 256);
            break;
        case 4: // Large All Particle Simulation
            setSimulationConfig(true, false, false, 65536, 65536, 65536);
            break;
        case 5: // Small Seeker Particle Simulation
            setSimulationConfig(false, false, false, 1000, 0, 0);
            break;
        case 6: // Large Seeker Particle Simulation
            setSimulationConfig(true, false, false, 100000, 0, 0);
            break;
        case 7: // Small Cruiser Particle Simulation
            setSimulationConfig(false, false, false, 0, 1000, 0);
            break;
        case 8: // Large Cruiser Particle Simulation
            setSimulationConfig(true, false, false, 0, 100000, 0);
            break;
        case 9: // Small Brownian Motion Simulation
            setSimulationConfig(false, false, false, 0, 0, 1000);
            break;
        case 10: // Large Brownian Motion Simulation
            setSimulationConfig(true, false, false, 0, 0, 100000);
            break;
        case 11: // Huge Seeker Particle Simulation
            setSimulationConfig(true, false, false, 5000000, 0, 0);
            break;
        default:
            std::cerr << "Error: --preset requires a positive integer argument from 1 to 13.\n";
            printUsage(execName);
            exit(1);
    }
}
void processCmdLineArgs(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--gpu-accel") == 0) {
            gpu_accel = true;
        } else if (strcmp(argv[i], "--stagger") == 0) {
            particle_stagger_start = true;
        } else if (strcmp(argv[i], "--can-die") == 0) {
            player_can_die = true;
        } else if (strcmp(argv[i], "--set-n") == 0) {
            if (i + 3 < argc) {
                n_seekers = std::atoi(argv[i + 1]);
                n_cruisers = std::atoi(argv[i + 2]);
                n_wanderers = std::atoi(argv[i + 3]);
                n_particles = n_seekers + n_cruisers + n_wanderers;
                i += 3; // Skip next three arguments as they are processed
            } else {
                std::cerr << "Error: --set-n requires three non-negative integer arguments.\n";
                printUsage(argv[0]);
                exit(1);
            }
        } else if (strcmp(argv[i], "--preset") == 0) {
            if (i + 1 < argc) {
                setSimulationPreset(std::atoi(argv[i + 1]), argv[0]);
                i++; // Skip next argument as it is processed
            } else {
                std::cerr << "Error: --preset requires a positive integer argument from 1 to 11.\n";
                printUsage(argv[0]);
                exit(1);
            }
        } else {
            std::cerr << "Error: Unknown option " << argv[i] << "\n";
            printUsage(argv[0]);
            exit(1);
        }
    }
}

// SCREEN AND GENERAL GAME CONSTANTS
////////////////////////////////////////////////////////////////////////////////
const char *SIMULATION_NAME = "Flicker";

// Window Dimensions are the same as Viewport Dimensions
const unsigned int WINDOW_WIDTH  = 800;
const unsigned int WINDOW_HEIGHT = 800;
const float SCALEX = 2.0f / WINDOW_WIDTH;
const float SCALEY = 2.0f / WINDOW_HEIGHT;
// Amount of extra space to render off screen (in NDC)
const float EXCESS_RENDER = 0.2f;

// All coordinates and measurements are stored in NDC
// Multiply by SCALEX and SCALEY to convert from pixel space to NDC


// PLAYER CONSTANTS
////////////////////////////////////////////////////////////////////////////////
const float PLAYER_WIDTH = 16.f * SCALEX;
const float PLAYER_HEIGHT = 24.f * SCALEY;
const float PLAYER_SPEED = 0.7f;
const float PLAYER_COLOR_FACTOR = 3.0f;


// PARTICLE CONSTANTS
////////////////////////////////////////////////////////////////////////////////
// how many particles of each type spawn each second
const float PARTICLE_STAGGER_PER_SEC = 1.5; // 3 
const float PARTICLE_POINT_SIZE = 7.f * SCALEX; // Square side-length in NDC
const float STARTING_BASE_SPEED = 0.086789f;
const float PARTICLE_BASE_SPEED_ACCEL = 0.000812345f;

// Seeker Specific Constants
// Each second seeker speed increases by this proportion of its inital velocity
// Note: This factor is multiplied by the starting x and y velocities respectively and 
// then kept constant. Acceleration must be proportional to x and y components
// of velocity to maintain the same direction 
const float SEEKER_ACCEL_FACTOR = 0.4f; 
const float SEEKER_COLOR_DIV_SCALE = 4.0f;

// Cruiser Specific Constants
// how much faster it is than particle base speed
const float CRUISER_FASTER_SPEED_RATIO = 3.f;
// how much distance to travel per checking if we should turn
const float CRUISER_CHECK_TURN_DIST = SCALEX; // setting to SCALE_X makes it checl once each pixel
// mutliplied by varying_pos to determine which decimal magnitude decides if we turn
const unsigned int CRUISER_DIST_SCALE = 1000; // setting 1000 has us check the thousands place
// controls turning freq, turns every TURNING_FREQ_CONSTANT checks 
const unsigned int CRUISER_TURNING_FREQ_INV = 499; 

// Wanderer Specific Constants
const float WANDERER_NO_INITIALIZATION_X = 0.2f;
const float WANDERER_NO_INITIALIZATION_Y = 0.2f;
const float WANDERER_FASTER_SPEED_RATIO = 1.0f;
const float WANDERER_BIGGER_SIZE_RATIO = 1.5f;



////////////////////////////////////////////////////////////////////////////////
// GPU COPIES OF RELEVANT PARTICLE CONSTANTS
////////////////////////////////////////////////////////////////////////////////
// General Constants
__constant__ float d_EXCESS_RENDER;
__constant__ float d_PARTICLE_POINT_SIZE;
__constant__ float d_STARTING_BASE_SPEED;
__constant__ float d_PARTICLE_BASE_SPEED_ACCEL;

// Seeker Specific Constants
__constant__ float d_SEEKER_ACCEL_FACTOR;
__constant__ float d_SEEKER_COLOR_DIV_SCALE;

// Cruiser Specific Constants
__constant__ float d_CRUISER_FASTER_SPEED_RATIO;
__constant__ float d_CRUISER_CHECK_TURN_DIST;
__constant__ unsigned int d_CRUISER_DIST_SCALE;
__constant__ unsigned int d_CRUISER_TURNING_FREQ_INV;

// Wanderer Specific Constants
__constant__ float d_WANDERER_NO_INITIALIZATION_X;
__constant__ float d_WANDERER_NO_INITIALIZATION_Y;
__constant__ float d_WANDERER_FASTER_SPEED_RATIO;
__constant__ float d_WANDERER_BIGGER_SIZE_RATIO;



// GPU CONSTANTS
////////////////////////////////////////////////////////////////////////////////
const unsigned int BLOCK_SIZE = 256;
const unsigned long long RANDOM_BASE_SEED = 1234;


// GENERAL GAME VARIABLES
////////////////////////////////////////////////////////////////////////////////
// Boolean for if display has started (Turned on by pressing space bar)
bool simulation_on = false;

// Variables for time
float totalTime = 0.0f;
float dt = 0.0f;
// Variables for FPS 
float timeOneSec = 0.0f;
float currentFPS = 0.0f;

// Used for keyboard handlers
// Keeps track of which keys are currently being pressed
std::map<unsigned char, bool> keyStates;
std::map<int, bool> specialKeyStates;


// GPU VARIABLES
////////////////////////////////////////////////////////////////////////////////
const unsigned int NUM_STREAMS = 3; // for number of particle types
cudaStream_t streams[NUM_STREAMS];
curandState *d_rand_states;
bool *d_player_death_occured;


// PARTICLE VARIABLES
////////////////////////////////////////////////////////////////////////////////
// VBO Variables
GLuint particleVBO;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;


// Field Variables
float particle_base_speed = STARTING_BASE_SPEED;
std::default_random_engine generator; // used for Brownian Motion

// Auto-Verification and FPS Vars
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;
// Timer is used to track FPS and time taken to calcute and render each frame,
// not for totalTime and dt (glutGet(GLUT_ELAPSED_TIME) is used for that)
StopWatchInterface *timer = NULL;



enum Direction { LEFT, RIGHT, DOWN, UP };


// PLAYER CLASS
////////////////////////////////////////////////////////////////////////////////
class Player {
private:
    // Fields
    ////////////////////////////////////////////////////////////////////////////
    float pos_x;
    float pos_y;

    float color_r;
    float color_g;
    float color_b;

    float vel_x;
    float vel_y;

    float width;
    float height;

    bool alive;

    // Private Methods
    ////////////////////////////////////////////////////////////////////////////
    // Checks which keys are currently being pressed to set velocity
    // Bounds players movement by restricting directional movement off-screen
    // Diagonal velocities are multiplied by sqrt(2)/2 for a consistent magnitude
    // Note: Top-level updateTime must be called first
    void updateVelocity() {
        vel_x = 0.0f;
        vel_y = 0.0f;

        bool up = keyStates['w'] || specialKeyStates[GLUT_KEY_UP];
        bool lt = keyStates['a'] || specialKeyStates[GLUT_KEY_LEFT];
        bool dn = keyStates['s'] || specialKeyStates[GLUT_KEY_DOWN];
        bool rt = keyStates['d'] || specialKeyStates[GLUT_KEY_RIGHT];

        if (!(up && dn)) {
            if (up) {
                if (pos_y + height/2.f < 1.0f)
                    vel_y = PLAYER_SPEED;
            } else if (dn) {
                if (pos_y - height/2.f > -1.0f)
                    vel_y = -PLAYER_SPEED;
            }
        }
        if (!(lt && rt)) {
            if (lt) {
                if (pos_x - width/2.f > -1.0f)
                    vel_x = -PLAYER_SPEED;
            } else if (rt) {
                if (pos_x + width/2.f < 1.0f)
                    vel_x = PLAYER_SPEED;
            }
        }

        if (vel_y != 0 && vel_x != 0) {
            vel_y *= SQRT2DIV2;
            vel_x *= SQRT2DIV2;
        }
    }

    // Basic position update based on velocity and dt
    // updateVelocity must be called first
    void updatePosition() {
        pos_x += vel_x * dt;
        pos_y += vel_y * dt;
    }

    // Updates color of the Player (smoothing over rainbow colors)
    // Note: Top-level updateTime must be called first
    void updateColor() {
        float pi_float = static_cast<float>(M_PI);
        color_r = (sinf(totalTime * PLAYER_COLOR_FACTOR) + 1.0f) / 2.0f;
        color_g = (sinf(totalTime * PLAYER_COLOR_FACTOR + 2.0f * pi_float / 3.0f) + 1.0f) / 2.0f;
        color_b = (sinf(totalTime * PLAYER_COLOR_FACTOR + 4.0f * pi_float / 3.0f) + 1.0f) / 2.0f;
    }

public:
    // Public Methods
    ////////////////////////////////////////////////////////////////////////////
    Player(float px, float py, float w, float h) {
        pos_x = px;
        pos_y = py;

        width = w;
        height = h;

        // Player starts alive
        alive = true;

        // Player has no initial velocity
        vel_x = 0.0f;
        vel_y = 0.0f;

        // Colors set through method based on total time
        updateColor();
    }      

    // Getters 
    float getPosX() {
        return pos_x;
    }
    float getPosY() {
        return pos_y;
    }
    float getWidth() {
        return width;
    }
    float getHeight() {
        return height;
    }
    bool isAlive() {
        return alive;
    }

    // Setters
    void kill() {
        if (player_can_die) {
            alive = false;

            // Effectively stops all behavior except player's color smoothing
            simulation_on = false;
        }
    }
    // Updates the vel, pos, and color of the Player (in that order)
    // Note: Top-level updateTime must be called first
    void update() {
        updateVelocity();
        updatePosition();
        updateColor();
    }
    // Updates only the color of the Player
    // Useful for rendering Player before starting simulation
    // Note: Top-level updateTime must be called first
    void updateOnlyColor() {
        updateColor();
    }


    // Make a different function for this so player doesn't need to import OpenGL
    void render() { 
        glBegin(GL_QUADS);
            glColor3f(color_r, color_g, color_b);
            glVertex2f(pos_x - width/ 2.f, pos_y - height / 2.f);  // Bottom left
            glVertex2f(pos_x + width/ 2.f, pos_y - height / 2.f);  // Bottom right
            glVertex2f(pos_x + width/ 2.f, pos_y + height / 2.f);  // Top right
            glVertex2f(pos_x - width/ 2.f, pos_y + height / 2.f);  // Top left
        glEnd();
    }
};


// PARTICLE STRUCTS
////////////////////////////////////////////////////////////////////////////////
/** Base Particle Struct 
 *  Has attributes of all Particles, all Particles inherit from this Struct 
 *  Conceptually, we opt to use Structs instead of Classes because we want to 
 *  implement the behavior-defining algorithms for all particles at once (and 
 *  in cuda), so there are no methods to declare.
 *
 *  This structure (and the optional attributes) makes it relatively easy to 
 *  define new particles without having to implement a class and virtual 
 *  methods (due to the reasons above).
 *  
 *  Currently, the Simulation supports 3 Particle Child Structs: 
 *       - Seeker
 *       - Cruiser
 *       - Wanderer
 *  These are described and implemented directly below.
 */
struct Particle {
    float pos_x;
    float pos_y;

    float color_r;
    float color_g;
    float color_b;

    float vel_x;
    float vel_y;

    float attr_a;
    float attr_b;
    // Add more additional attributes as necessary
};

/** Seeker (Particle 1)
 *  This particle aims at the Player's position (at spawn time) 
 *  and darts straight there with increasing acceleration.
 *  Additional attributes:
 *  attr_a : accel_x
 *  attr_b : accel_y
 *  Component acceleration = component vel * factor to maintain direction of motion
 *  This particle also changes colors from red to yellow depending on much how
 *  faster it is, computed as a ratio of current speed to starting base speed
 */ 
struct Seeker   : Particle { };

/** Cruiser (Particle 2)
 *  This particle shoots in a simple (horizontal or vertical) direction 
 *  but turns at seemingly random times.
 *  Additional attributes:
 *  attr_a : distance_traveled_since_last_check
 *  This particle also changes colors depending the direction it's going,
 *  Horizontally moving particles are Green, vertically moving particles are Violet
 *  Psuedo-Random Frame-Indepedent Turning:
 *    There are 2 challenges with designing this behavior:
 *      - Want random turning w/out needing to keep regenerating random numbers
 *      - Need it to be frame-independent so different frame rates don't result in different behavior 
 *    Procedure: 
 *      In update, each time distance_traveled_since_last_check exceeds CRUISER_CHECK_TURN_DIST:
 *          Turn if ((int)(varying_position * 1000) % TURNING_FREQ_CONSTANT == 0)
 *              If turning, also use varying_pos to determine direction
 */ 
struct Cruiser  : Particle { };

/** Wanderer (Particle 3)
 *  This particle wanders around by simulating Brownian motion, creating hot 
 *  spots of dangerous unpredictability for the Player.
 *  Brownian motion models the random motion of particles suspended in a medium 
 *  and is a mean zero, continuous process, often implemented using GPU 
 *  acceleration due to its computational parallelism.
 *  Additional attributes: None
 *  This particle is gray colored.
 */ 
struct Wanderer : Particle { };


// Players
////////////////////////////////////////////////////////////////////////////////
Player *p1;


// Declarations
////////////////////////////////////////////////////////////////////////////////
// Set up and clean up
bool runSimulation(int argc, char **argv);

// GL functionality
bool initGL(int *argc, char **argv);
void createParticleVBO(GLuint *vbo);
void deleteParticleVBO(GLuint *vbo);

// Declaring Callbacks and Handlers
void display();
void redisplay();
void cleanup();
void handleKeyPress(unsigned char key, int x, int y);
void handleKeyRelease(unsigned char key, int x, int y);
void handleSpecialKeyPress(int key, int x, int y);
void handleSpecialKeyRelease(int key, int x, int y);

// Display callback helper functions needed for display
void updateTime();
void computeFPS();

// Utility functions for generating a random numbers on the CPU
float randomFloatCPU(float min, float max); // range: [min, max)
int randomIntCPU(int min, int max); // range: [min, max]


////////////////////////////////////////////////////////////////////////////////
//! Particle Helper Functions Defined on both the CPU and GPU
////////////////////////////////////////////////////////////////////////////////
// Returns true if the particle has collided with the player
__host__ __device__ bool detectPlayerCollision(float px, float py, float psize,
        float target_x, float target_y, float target_w, float target_h) {
    /// Calculates the edges of the particle
    float halfLength = psize / 2.f;
    float left = px - halfLength;
    float right = px + halfLength;
    float top = py + halfLength;
    float bottom = py - halfLength;

    // Calculates the edges of the player
    float targetLeft = target_x - target_w / 2.0f;
    float targetRight = target_x + target_w / 2.0f;
    float targetTop = target_y + target_h / 2.0f;
    float targetBottom = target_y - target_h / 2.0f;

    // Checks for overlap
    return (left < targetRight && right > targetLeft
        && top > targetBottom && bottom < targetTop);
}

// Returns true if the particle is outside the screen bounds
// (including the excess render region)
__host__ __device__ bool detectOffScreen(float px, float py, 
        float psize, float excess_render) {
    // Calculates the edges of the particle
    float halfLength = psize / 2.f;
    float left = px - halfLength;
    float right = px + halfLength;
    float top = py + halfLength;
    float bottom = py - halfLength;

    float screen_min = -1.f - excess_render;
    float screen_max = 1.f + excess_render;

    // Checks if the particle is outside screen boundaries
    return (left < screen_min || right > screen_max
        || top > screen_max || bottom < screen_min);
}


////////////////////////////////////////////////////////////////////////////////
//! GPU Particle Functions
////////////////////////////////////////////////////////////////////////////////
int getNumBlocks(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


//  GPU Set Up and Clean Up Functions
////////////////////////////////////////////////////////////////////////////////
void copyConstantsToGPU() {
    // Note: These don't need to be freed bc they are implicitly managed by CUDA runtime
    cudaMemcpyToSymbol(d_EXCESS_RENDER, &EXCESS_RENDER, sizeof(float));
    cudaMemcpyToSymbol(d_PARTICLE_POINT_SIZE, &PARTICLE_POINT_SIZE, sizeof(float));
    cudaMemcpyToSymbol(d_STARTING_BASE_SPEED, &STARTING_BASE_SPEED, sizeof(float));
    cudaMemcpyToSymbol(d_PARTICLE_BASE_SPEED_ACCEL, &PARTICLE_BASE_SPEED_ACCEL, sizeof(float));

    cudaMemcpyToSymbol(d_SEEKER_ACCEL_FACTOR, &SEEKER_ACCEL_FACTOR, sizeof(float));
    cudaMemcpyToSymbol(d_SEEKER_COLOR_DIV_SCALE, &SEEKER_COLOR_DIV_SCALE, sizeof(float));

    cudaMemcpyToSymbol(d_CRUISER_FASTER_SPEED_RATIO, &CRUISER_FASTER_SPEED_RATIO, sizeof(float));
    cudaMemcpyToSymbol(d_CRUISER_CHECK_TURN_DIST, &CRUISER_CHECK_TURN_DIST, sizeof(float));
    cudaMemcpyToSymbol(d_CRUISER_DIST_SCALE, &CRUISER_DIST_SCALE, sizeof(unsigned int));
    cudaMemcpyToSymbol(d_CRUISER_TURNING_FREQ_INV, &CRUISER_TURNING_FREQ_INV, sizeof(unsigned int));

    cudaMemcpyToSymbol(d_WANDERER_NO_INITIALIZATION_X, &WANDERER_NO_INITIALIZATION_X, sizeof(float));
    cudaMemcpyToSymbol(d_WANDERER_NO_INITIALIZATION_Y, &WANDERER_NO_INITIALIZATION_Y, sizeof(float));
    cudaMemcpyToSymbol(d_WANDERER_FASTER_SPEED_RATIO, &WANDERER_FASTER_SPEED_RATIO, sizeof(float));
    cudaMemcpyToSymbol(d_WANDERER_BIGGER_SIZE_RATIO, &WANDERER_BIGGER_SIZE_RATIO, sizeof(float));
}
// CUDA kernel to initialize the random states
__global__ void init_rng_states(curandState *states, int n, int base_seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        curand_init(base_seed + id, id, 0, &states[id]);
    }
}
void setUpGPU(GLuint *p_vbo, struct cudaGraphicsResource **vbo_res, 
        unsigned int vbo_res_flags) 
{
    // Copies all the constants on the host to the GPU constant memory
    copyConstantsToGPU();

    // Registers Particle VBO buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *p_vbo, vbo_res_flags));

    // Creates all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    // Initializes Random Number Generator for each particle
    checkCudaErrors( cudaMalloc(&d_rand_states, n_particles * sizeof(curandState)) );
    init_rng_states<<<getNumBlocks(n_particles), BLOCK_SIZE>>>(
        d_rand_states, n_particles, RANDOM_BASE_SEED);

    // Initializes array to track player death in the GPU update function
    checkCudaErrors(cudaMalloc((void**)&d_player_death_occured, 3 * sizeof(bool)));
}
void cleanUpGPU(struct cudaGraphicsResource *vbo_res) {
    // Unregisters this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    checkCudaErrors(cudaFree(d_rand_states));
    checkCudaErrors(cudaFree(d_player_death_occured));
}



//  GPU Random Functions
////////////////////////////////////////////////////////////////////////////////
// Device function to generate a random int between a and b (both inclusive)
__device__ int randomIntGPU(curandState *state, int a, int b) {
    float myrandf = curand_uniform(state);  // Uniform random float between 0 and 1
    return a + (int)((b - a + 1) * myrandf); // Scale and shift to [a, b]
}
// Device function to generate a random float between a and b (both inclusive)
__device__ float randomFloatGPU(curandState *state, float a, float b) {
    float myrandf = curand_uniform(state);  // Uniform random float between 0 and 1
    return a + (b - a) * myrandf;           // Scale and shift to [a, b]
}
// Device function to generate a random starting position and return the direction
__device__ Direction randomStartPosGPU(curandState *state, float* pos_x, float *pos_y) {
    Direction dir = static_cast<Direction>(randomIntGPU(state, 0, 3));

    switch (dir) {
        case LEFT: // Left edge
            *pos_x = -1.f - randomFloatGPU(state, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER);
            *pos_y = randomFloatGPU(state, -1.f, 1.f);
            break;
        case RIGHT: // Right edge
            *pos_x = 1.f + randomFloatGPU(state, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER);
            *pos_y = randomFloatGPU(state, -1.f, 1.f);
            break;
        case DOWN: // Bottom edge
            *pos_x = randomFloatGPU(state, -1.f, 1.f);
            *pos_y = -1.f - randomFloatGPU(state, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER);
            break;
        case UP: // Top edge
            *pos_x = randomFloatGPU(state, -1.f, 1.f);
            *pos_y = 1.f + randomFloatGPU(state, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER);
            break;
    }
    return dir;
}



//  GPU Seeker Functions
////////////////////////////////////////////////////////////////////////////////
__device__ void deviceInitSeeker(curandState *state, Seeker *s, 
        float particle_base_speed, float target_pos_x, float target_pos_y) 
{
    // Color (set to red, green changes in updateColor to get yellow)
    s->color_r = 1.0f;
    s->color_g = 0.0f;
    s->color_b = 0.0f;

    // Position
    randomStartPosGPU(state, &(s->pos_x), &(s->pos_y));

    // Velocity
    float dx = target_pos_x - s->pos_x;
    float dy = target_pos_y - s->pos_y;
    float dist = sqrtf(dx * dx + dy * dy);
    float ux = dx / dist;
    float uy = dy / dist;
    s->vel_x = ux * particle_base_speed;
    s->vel_y = uy * particle_base_speed;

    // Additional Attribute: Acceleration
    s->attr_a = s->vel_x * d_SEEKER_ACCEL_FACTOR;
    s->attr_b = s->vel_y * d_SEEKER_ACCEL_FACTOR;
}
__global__ void kernelInitSeekers(curandState *states, Seeker *seekers, int n, 
        float particle_base_speed, float target_pos_x, float target_pos_y) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Seeker *s = &(seekers[idx]);
    curandState *localState = &(states[idx]);

    deviceInitSeeker(localState, s, particle_base_speed, target_pos_x, target_pos_y);
}
__global__ void kernelUpdateSeekers(curandState *states, Seeker *seekers, int n, 
        float particle_base_speed, bool *target_death, 
        float dt, bool staggerStart, int waitUntilBound,
        float target_x, float target_y, float target_w, float target_h) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Staggers starts when toggled on (for only Seeker and Cruiser)
    if (staggerStart && idx > waitUntilBound) return;

    Seeker *s = &(seekers[idx]);
    curandState *state = &(states[idx]);

    // Basic Velocity Update
    s->vel_x += s->attr_a * dt;
    s->vel_y += s->attr_b * dt;

    // Basic Position Update
    s->pos_x += s->vel_x * dt;
    s->pos_y += s->vel_y * dt;

    // Color Update: Hotter Effect
    // Increases the green value to make color go from red to yellow 
    // the faster it gets, simulating a "hotter", "more firey" look
    float speed = sqrtf(s->vel_x * s->vel_x + s->vel_y * s->vel_y);
    float ratioToStartingSpeed = speed / d_STARTING_BASE_SPEED - 1.f;
    s->color_g = (ratioToStartingSpeed - 1.f) / d_SEEKER_COLOR_DIV_SCALE;    

    // Do player collision first in case particle has collided with 
    // player a little outside of the screen bounds

    // Player Collision
    if (detectPlayerCollision(s->pos_x, s->pos_y, d_PARTICLE_POINT_SIZE,
            target_x, target_y, target_w, target_h)) {
        *target_death = true;
    }

    // If Offscreen -> Respawn
    if (detectOffScreen(s->pos_x, s->pos_y, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER)) {
        deviceInitSeeker(state, s, particle_base_speed, target_x, target_y);
    }
}



//  GPU Cruiser Functions
////////////////////////////////////////////////////////////////////////////////
__device__ void deviceInitCruiser(curandState *state, Cruiser *c, 
        float particle_base_speed) 
{
    Direction edge = randomStartPosGPU(state, &(c->pos_x), &(c->pos_y));

    // Velocity and Color
    c->vel_x = 0.0f;  
    c->vel_y = 0.0f;
    c->color_r = 0.0f;   // Sets horizontally moving to green
    c->color_g = 0.0f;   // Sets vertically moving to violet
    c->color_b = 0.0f;
    switch (edge) {
        case LEFT: // Left edge
            c->vel_x = particle_base_speed * d_CRUISER_FASTER_SPEED_RATIO;
            c->color_g = 1.0f;
            break;
        case RIGHT: // Right edge
            c->vel_x = -particle_base_speed * d_CRUISER_FASTER_SPEED_RATIO;
            c->color_g = 1.0f;
            break;
        case DOWN: // Bottom edge
            c->vel_y = particle_base_speed *  d_CRUISER_FASTER_SPEED_RATIO;
            c->color_r = 1.0f;
            c->color_b = 1.0f;
            break;
        case UP: // Top edge
            c->vel_y = -particle_base_speed * d_CRUISER_FASTER_SPEED_RATIO;
            c->color_r = 1.0f;
            c->color_b = 1.0f;
            break;
    }

    // Additional Attribute: Distance Traveled Since Last Check
    c->attr_a = 0.0f;
}
__global__ void kernelInitCruisers(curandState *states, Cruiser *cruisers, int n, 
        float particle_base_speed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Cruiser *c = &(cruisers[idx]);
    curandState *localState = &(states[idx]);

    deviceInitCruiser(localState, c, particle_base_speed);
}
__global__ void kernelUpdateCruisers(curandState *states, Cruiser *cruisers, int n, 
        float particle_base_speed, bool *target_death, 
        float dt, bool staggerStart, int waitUntilBound,
        float target_x, float target_y, float target_w, float target_h) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Staggers starts when toggled on (for only Seeker and Cruiser)
    if (staggerStart && idx > waitUntilBound) return;
    
    Cruiser *c = &(cruisers[idx]);
    curandState *state = &(states[idx]);

    // Position and Distance Traveled Since Last Check Update
    bool horizontal = (c->vel_x != 0.0f);
    if (horizontal) {
        float dx = c->vel_x * dt;
        c->pos_x += dx;
        c->attr_a += fabsf(dx);
    } else {
        float dy = c->vel_y * dt;
        c->pos_y += dy;
        c->attr_a += fabsf(dy);
    }

    // Psuedo-Random Frame-Indepedent Turning
    if (c->attr_a > d_CRUISER_CHECK_TURN_DIST) { // Then checks if we should turn
        c->attr_a = 0.0f; // Resets Distance Traveled Since Last Check
        float varying_pos = horizontal ? c->pos_x : c->pos_y;
        int turn_decider = (int)(varying_pos * d_CRUISER_DIST_SCALE);
        if (turn_decider % d_CRUISER_TURNING_FREQ_INV == 0) {
            float dir_change = ((turn_decider/10) % 2 == 0) ? -1.f : 1.f;
            if (horizontal) {
                // Speed Reversal (Horizontal to Vertical)
                c->vel_y = c->vel_x * dir_change;
                c->vel_x = 0.0f;
                // Color Change (Green to Violet)
                c->color_r = 1.0f;
                c->color_g = 0.0f;
                c->color_b = 1.0f;
            } else {
                // Speed Reversal (Vertical to Horizontal)
                c->vel_x = c->vel_y * dir_change;
                c->vel_y = 0.0f;
                // Color Change (Violet to Green)
                c->color_r = 0.0f;
                c->color_g = 1.0f;
                c->color_b = 0.0f;
            }
        }
    }

    // Do player collision first in case particle has collided with 
    // player a little outside of the screen bounds

    // Player Collision
    if (detectPlayerCollision(c->pos_x, c->pos_y, d_PARTICLE_POINT_SIZE,
            target_x, target_y, target_w, target_h)) {
        *target_death = true;
    }

    // If Offscreen -> Respawn
    if (detectOffScreen(c->pos_x, c->pos_y, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER)) {
        deviceInitCruiser(state, c, particle_base_speed);
    }
}



//  GPU Wanderer Functions
////////////////////////////////////////////////////////////////////////////////
__device__ void deviceInitWanderer(curandState *state, Wanderer *w, bool respawn,
        float particle_base_speed) 
{
    // Color (set to light gray)
    w->color_r = 0.8f;
    w->color_g = 0.8f;
    w->color_b = 0.8f;

    // Position
    if (respawn) {
        randomStartPosGPU(state, &(w->pos_x), &(w->pos_y));
    } else {
        w->pos_x = randomFloatGPU(state, d_WANDERER_NO_INITIALIZATION_X, 1.f + d_EXCESS_RENDER);
        w->pos_y = randomFloatGPU(state, d_WANDERER_NO_INITIALIZATION_Y, 1.f + d_EXCESS_RENDER);
        w->pos_x *= randomIntGPU(state, 0, 1) ? -1.f : 1.f;
        w->pos_y *= randomIntGPU(state, 0, 1) ? -1.f : 1.f;
    }

    // Velocity
    w->vel_x = particle_base_speed * d_WANDERER_FASTER_SPEED_RATIO;
    w->vel_y = particle_base_speed * d_WANDERER_FASTER_SPEED_RATIO;

    // Additional Attribute: None
}
__global__ void kernelInitWanderers(curandState *states, Wanderer *wanderers, int n, 
        float particle_base_speed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Wanderer *w = &(wanderers[idx]);
    curandState *localState = &(states[idx]);

    deviceInitWanderer(localState, w, false, particle_base_speed);
}
__global__ void kernelUpdateWanderers(curandState *states, Wanderer *wanderers, int n, 
    float particle_base_speed, bool *target_death, float dt, 
    float target_x, float target_y, float target_w, float target_h) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Wanderers don't stagger starts bc they spawn on screen

    Wanderer *w = &(wanderers[idx]);
    curandState *state = &(states[idx]);

    // Position Update (Brownian Motion Implementation)
    w->pos_x += curand_normal(state) * sqrtf(dt) * w->vel_x;
    w->pos_y += curand_normal(state) * sqrtf(dt) * w->vel_y;

    // Do player collision first in case particle has collided with 
    // player a little outside of the screen bounds

    // Player Collision
    if (detectPlayerCollision(w->pos_x, w->pos_y, 
            d_PARTICLE_POINT_SIZE * d_WANDERER_BIGGER_SIZE_RATIO,
            target_x, target_y, target_w, target_h)) {
        *target_death = true;
    }

    // If Offscreen -> Respawn
    if (detectOffScreen(w->pos_x, w->pos_y, d_PARTICLE_POINT_SIZE, d_EXCESS_RENDER)) {
        deviceInitWanderer(state, w, true, particle_base_speed);
    }
}



//  Main GPU Particle Functions
////////////////////////////////////////////////////////////////////////////////
void initParticlesGPU(struct cudaGraphicsResource **vbo_resource) {
    cudaDeviceSynchronize();

    // Maps the Particle VBO for writing from CUDA
    Particle *dvbo;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dvbo, &num_bytes,
                                                         *vbo_resource));

    // Initializes all particles in the VBO                                           
    Seeker *d_seekers = static_cast<Seeker*>(dvbo);
    kernelInitSeekers<<<getNumBlocks(n_seekers), BLOCK_SIZE, 0, streams[0]>>>(
        d_rand_states, d_seekers, n_seekers, 
        particle_base_speed, p1->getPosX(), p1->getPosY());

    Cruiser *d_cruisers = static_cast<Cruiser*>(dvbo+n_seekers);
    kernelInitCruisers<<<getNumBlocks(n_cruisers), BLOCK_SIZE, 0, streams[1]>>>(
        d_rand_states+n_seekers, d_cruisers, n_cruisers,
        particle_base_speed);

    Wanderer *d_wanderers = static_cast<Wanderer*>(dvbo+n_seekers+n_cruisers);
    kernelInitWanderers<<<getNumBlocks(n_wanderers), BLOCK_SIZE, 0, streams[2]>>>(
        d_rand_states+n_seekers+n_cruisers, d_wanderers, n_wanderers,
        particle_base_speed);

    // Unmaps the Particle VBO
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}
void updateParticlesGPU(struct cudaGraphicsResource **vbo_resource) {
    cudaDeviceSynchronize();

    // Maps the Particle VBO for writing from CUDA
    Particle *dvbo;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dvbo, &num_bytes,
                                                         *vbo_resource));                                                        

    // Used to stagger starts if toggled on through particle_stagger_start
    int waitUntilBound = static_cast<int>(totalTime * PARTICLE_STAGGER_PER_SEC);

    /* Make a separate player death bool for each stream to prevent sequential access
     * It's okay for mutiple threads in the same stream to access an element in 
     * d_player_death_occured bc they would both be changing it to the same value
     * bring true, so there's no race condition.
     */
    checkCudaErrors(cudaMemset(d_player_death_occured, 0, 3 * sizeof(bool)));


    // Updates all particles in the VBO 
    Seeker *d_seekers = static_cast<Seeker*>(dvbo);
    kernelUpdateSeekers<<<getNumBlocks(n_seekers), BLOCK_SIZE, 0, streams[0]>>>(
        d_rand_states, d_seekers, n_seekers, 
        particle_base_speed, d_player_death_occured, 
        dt, particle_stagger_start, waitUntilBound, 
        p1->getPosX(), p1->getPosY(), p1->getWidth(), p1->getHeight());
            
    Cruiser *d_cruisers = static_cast<Cruiser*>(dvbo+n_seekers);
    kernelUpdateCruisers<<<getNumBlocks(n_cruisers), BLOCK_SIZE, 0, streams[1]>>>(
        d_rand_states+n_seekers, d_cruisers, n_cruisers,
        particle_base_speed, d_player_death_occured + 1, 
        dt, particle_stagger_start, waitUntilBound, 
        p1->getPosX(), p1->getPosY(), p1->getWidth(), p1->getHeight());

    Wanderer *d_wanderers = static_cast<Wanderer*>(dvbo+n_seekers+n_cruisers);
    kernelUpdateWanderers<<<getNumBlocks(n_wanderers), BLOCK_SIZE, 0, streams[2]>>>(
        d_rand_states+n_seekers+n_cruisers, d_wanderers, n_wanderers,
        particle_base_speed, d_player_death_occured + 2, dt,
        p1->getPosX(), p1->getPosY(), p1->getWidth(), p1->getHeight());
    

    // Kills player if any of the particles (in any of the 3 streams) collided with the player
    cudaDeviceSynchronize();
    bool player_death_occured[3];
    checkCudaErrors(cudaMemcpy(player_death_occured, d_player_death_occured, 3 * sizeof(bool), 
        cudaMemcpyDeviceToHost));
    if (player_death_occured[0] || player_death_occured[1] || player_death_occured[2]) {
        p1->kill();
    }

    // Unmaps the Particle VBO
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}




////////////////////////////////////////////////////////////////////////////////
//! CPU Particle Functions
////////////////////////////////////////////////////////////////////////////////
// Starts PARTICLE_STAGGER_PER_SEC of each particle per second if particle_stagger_start is on
bool shouldDelayStart(int particleIdx) {
    if (!particle_stagger_start) {
        return false;
    }
    int waitUntilBound = static_cast<int>(totalTime * PARTICLE_STAGGER_PER_SEC); 
    return (particleIdx > waitUntilBound);
}

Direction randStartPosCPU(float &pos_x, float &pos_y);

void initSeekerCPU(Seeker &s);
void updateSeekerCPU(Seeker &s);

void initCruiserCPU(Cruiser &c);
void updateCruiserCPU(Cruiser &c);

void initWandererCPU(Wanderer &w, bool respawn);
void updateWandererCPU(Wanderer &w, std::normal_distribution<float> distribution);


bool initParticlesCPU() {
    // Maps the Particle VBO in order to modify it directly
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    Particle* ptr = (Particle*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    if (!ptr) {
        fprintf(stderr, "ERROR: Failed to Map Particle Buffer for CPU Initialization.");
        fflush(stderr);
        g_TotalErrors++;
        return false;
    }

    unsigned int i = 0;
    while (i < n_seekers) {
        initSeekerCPU( ((Seeker*)ptr)[i++] );
    }
    while (i < n_seekers + n_cruisers) {
        initCruiserCPU( ((Cruiser*)ptr)[i++] );
    }
    while (i < n_seekers + n_cruisers + n_wanderers) {
        initWandererCPU( ((Wanderer*)ptr)[i++], false );
    }

    glUnmapBuffer(GL_ARRAY_BUFFER); // Unmaps the buffer when done
    return true;   
}


bool updateParticlesCPU() {
    // Maps the Particle VBO in order to modify it directly
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    Particle* ptr = (Particle*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    if (!ptr) {
        fprintf(stderr, "ERROR: Failed to Map Particle Buffer for CPU Update.");
        fflush(stderr);
        g_TotalErrors++;
        return false;
    }
    
    unsigned int i = 0;
    // Updates Seeker Particles
    while (i < n_seekers) {
        if (shouldDelayStart(i)) {
            i++;
            continue;
        }
        updateSeekerCPU( ((Seeker*)ptr)[i++] );
    }
    // Updates Cruiser Particles
    while (i < n_seekers + n_cruisers) {
        if (shouldDelayStart(i - n_seekers)) {
            i++;
            continue;
        }
        updateCruiserCPU( ((Cruiser*)ptr)[i++] );
    }
    // Updates Wanderer Particles
    // In the rare case the sqrt(dt) is too small, we skip the update
    bool skipUpdate = false;
    float stddev = sqrtf(dt);
    if (stddev <= 0) {
        skipUpdate = true;
    }
    if (!skipUpdate) {
        std::normal_distribution<float> distribution(0.0f, stddev);
        while (i < n_seekers + n_cruisers + n_wanderers) {
            // No updates delay for wanderers since they are already on screen
            updateWandererCPU( ((Wanderer*)ptr)[i++], distribution );
        }
    }

    glUnmapBuffer(GL_ARRAY_BUFFER); // Unmaps the buffer when done
    return true; 
}



//  CPU Seeker Functions
////////////////////////////////////////////////////////////////////////////////
void initSeekerCPU(Seeker &s) {
    // Color (set to red, green changes in updateColor to get yellow)
    s.color_r = 1.0f;
    s.color_g = 0.0f;
    s.color_b = 0.0f;

    // Position
    randStartPosCPU(s.pos_x, s.pos_y);

    // Velocity
    float dx = p1->getPosX() - s.pos_x;
    float dy = p1->getPosY() - s.pos_y;
    float dist = sqrtf(dx * dx + dy * dy);
    float ux = dx / dist;
    float uy = dy / dist;
    s.vel_x = ux * particle_base_speed;
    s.vel_y = uy * particle_base_speed;

    // Additional Attribute: Acceleration
    s.attr_a = s.vel_x * SEEKER_ACCEL_FACTOR;
    s.attr_b = s.vel_y * SEEKER_ACCEL_FACTOR;
}


void updateSeekerCPU(Seeker &s) {
    // Basic Velocity Update
    s.vel_x += s.attr_a * dt;
    s.vel_y += s.attr_b * dt;

    // Basic Position Update
    s.pos_x += s.vel_x * dt;
    s.pos_y += s.vel_y * dt;

    // Color Update: Hotter Effect
    // Increases the green value to make color go from red to yellow 
    // the faster it gets, simulating a "hotter", "more firey" look
    float speed = sqrtf(s.vel_x * s.vel_x + s.vel_y * s.vel_y);
    float ratioToStartingSpeed = speed / STARTING_BASE_SPEED - 1.f;
    s.color_g = (ratioToStartingSpeed - 1.f) / SEEKER_COLOR_DIV_SCALE;

    // Do player collision first in case particle has collided with 
    // player a little outside of the screen bounds

    // Player Collision
    if (detectPlayerCollision(s.pos_x, s.pos_y, PARTICLE_POINT_SIZE,
            p1->getPosX(), p1->getPosY(), p1->getWidth(), p1->getHeight())) {
        p1->kill();
    }

    // If Offscreen -> Respawn
    if (detectOffScreen(s.pos_x, s.pos_y, PARTICLE_POINT_SIZE, EXCESS_RENDER)) {
        initSeekerCPU(s);
    }
}



//  CPU Cruiser Functions
////////////////////////////////////////////////////////////////////////////////
void initCruiserCPU(Cruiser &c) {
    // Position
    Direction edge = randStartPosCPU(c.pos_x, c.pos_y);

    // Velocity and Color
    c.vel_x = 0.0f;  
    c.vel_y = 0.0f;
    c.color_r = 0.0f;   // Sets horizontally moving to green
    c.color_g = 0.0f;   // Sets vertically moving to violet
    c.color_b = 0.0f;
    switch (edge) {
        case LEFT: // Left edge
            c.vel_x = particle_base_speed * CRUISER_FASTER_SPEED_RATIO;
            c.color_g = 1.0f;
            break;
        case RIGHT: // Right edge
            c.vel_x = -particle_base_speed * CRUISER_FASTER_SPEED_RATIO;
            c.color_g = 1.0f;
            break;
        case DOWN: // Bottom edge
            c.vel_y = particle_base_speed *  CRUISER_FASTER_SPEED_RATIO;
            c.color_r = 1.0f;
            c.color_b = 1.0f;
            break;
        case UP: // Top edge
            c.vel_y = -particle_base_speed * CRUISER_FASTER_SPEED_RATIO;
            c.color_r = 1.0f;
            c.color_b = 1.0f;
            break;
    }

    // Additional Attribute: Distance Traveled Since Last Check
    c.attr_a = 0.0f;
}


void updateCruiserCPU(Cruiser &c) {
    // Position and Distance Traveled Since Last Check Update
    bool horizontal = (c.vel_x != 0.0f);
    if (horizontal) {
        float dx = c.vel_x * dt;
        c.pos_x += dx;
        c.attr_a += fabsf(dx);
    } else {
        float dy = c.vel_y * dt;
        c.pos_y += dy;
        c.attr_a += fabsf(dy);
    }

    // Psuedo-Random Frame-Indepedent Turning
    if (c.attr_a > CRUISER_CHECK_TURN_DIST) { // Then checks if we should turn
        c.attr_a = 0.0f; // Resets Distance Traveled Since Last Check
        float varying_pos = horizontal ? c.pos_x : c.pos_y;
        int turn_decider = (int)(varying_pos * CRUISER_DIST_SCALE);
        if (turn_decider % CRUISER_TURNING_FREQ_INV == 0) {
            float dir_change = ((turn_decider/10) % 2 == 0) ? -1.f : 1.f;
            if (horizontal) {
                // Speed Reversal (Horizontal to Vertical)
                c.vel_y = c.vel_x * dir_change;
                c.vel_x = 0.0f;
                // Color Change (Green to Violet)
                c.color_r = 1.0f;
                c.color_g = 0.0f;
                c.color_b = 1.0f;
            } else {
                // Speed Reversal (Vertical to Horizontal)
                c.vel_x = c.vel_y * dir_change;
                c.vel_y = 0.0f;
                // Color Change (Violet to Green)
                c.color_r = 0.0f;
                c.color_g = 1.0f;
                c.color_b = 0.0f;
            }
        }
    }
    
    // Do player collision first in case particle has collided with 
    // player a little outside of the screen bounds

    // Player Collision
    if (detectPlayerCollision(c.pos_x, c.pos_y, PARTICLE_POINT_SIZE,
            p1->getPosX(), p1->getPosY(), p1->getWidth(), p1->getHeight())) {
        p1->kill();
    }

    // If Offscreen -> Respawn
    if (detectOffScreen(c.pos_x, c.pos_y, PARTICLE_POINT_SIZE, EXCESS_RENDER)) {
        initCruiserCPU(c);
    }
}



//  CPU Wanderer Functions
////////////////////////////////////////////////////////////////////////////////
void initWandererCPU(Wanderer &w, bool respawn) {
    // Color (set to light gray)
    w.color_r = 0.8f;
    w.color_g = 0.8f;
    w.color_b = 0.8f;

    // Position
    if (respawn) {
        randStartPosCPU(w.pos_x, w.pos_y);
    } else {
        w.pos_x = randomFloatCPU(WANDERER_NO_INITIALIZATION_X, 1.f + EXCESS_RENDER);
        w.pos_y = randomFloatCPU(WANDERER_NO_INITIALIZATION_Y, 1.f + EXCESS_RENDER);
        w.pos_x *= randomIntCPU(0, 1) ? -1.f : 1.f;
        w.pos_y *= randomIntCPU(0, 1) ? -1.f : 1.f;
    }

    // Velocity
    w.vel_x = particle_base_speed * WANDERER_FASTER_SPEED_RATIO;
    w.vel_y = particle_base_speed * WANDERER_FASTER_SPEED_RATIO;

    // Additional Attribute: None
}


void updateWandererCPU(Wanderer &w, std::normal_distribution<float> distribution) {
    // Doesn't use dt bc it's encoded into the creation of the distribution

    // Position Update (Brownian Motion Implementation)
    w.pos_x += distribution(generator) * w.vel_x;
    w.pos_y += distribution(generator) * w.vel_y;

    // Do player collision first in case particle has collided with 
    // player a little outside of the screen bounds

    // Player Collision
    if (detectPlayerCollision(w.pos_x, w.pos_y, 
            PARTICLE_POINT_SIZE * WANDERER_BIGGER_SIZE_RATIO,
            p1->getPosX(), p1->getPosY(), p1->getWidth(), p1->getHeight())) {
        p1->kill();
    }

    // If Offscreen -> Respawn
    if (detectOffScreen(w.pos_x, w.pos_y, 
            PARTICLE_POINT_SIZE * WANDERER_BIGGER_SIZE_RATIO, EXCESS_RENDER)) {
        // Respawns wanderers on the edge of the map instead of somewhere in the middle
        initWandererCPU(w, true);
    }
}



//  CPU Random Utility Functions
////////////////////////////////////////////////////////////////////////////////

// Generates a random starting position (in NDC, not pixels) somewhere on the 
// boundary of the screen by generating 3 random numbers (1 int, 2 floats).
// Returns which edge it placed the point at as a Direction
// One coordinate within EXCESS_RENDER region and other is within screen bounds
// Note: position is given in NDC
Direction randStartPosCPU(float &pos_x, float &pos_y) {
    Direction dir = static_cast<Direction>(randomIntCPU(0, 3));

    switch (dir) {
        case LEFT: // Left edge
            pos_x = -1.f - randomFloatCPU(PARTICLE_POINT_SIZE,EXCESS_RENDER);
            pos_y = randomFloatCPU(-1.f, 1.f);
            break;
        case RIGHT: // Right edge
            pos_x = 1.f + randomFloatCPU(PARTICLE_POINT_SIZE, EXCESS_RENDER);
            pos_y = randomFloatCPU(-1.f, 1.f);
            break;
        case DOWN: // Bottom edge
            pos_x = randomFloatCPU(-1.f, 1.f);
            pos_y = -1.f - randomFloatCPU(PARTICLE_POINT_SIZE,EXCESS_RENDER);
            break;
        case UP: // Top edge
            pos_x = randomFloatCPU(-1.f, 1.f);
            pos_y = 1.f + randomFloatCPU(PARTICLE_POINT_SIZE,EXCESS_RENDER);
            break;
    }
    return dir;
}

////////////////////////////////////////////////////////////////////////////////
// Generates a random float between min (inclusive) and max (exclusive)
float randomFloatCPU(float min, float max) {
    // Creates a random device and uses it to seed the RNG
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution range [min, max)
    std::uniform_real_distribution<float> dis(min, max);

    // Generate and return the random float
    return dis(gen);
}

// Generates a random int between min (inclusive) and max (inclusive)
int randomIntCPU(int min, int max) {
    // Creates a random device and uses it to seed the RNG
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution range [min, max]
    std::uniform_int_distribution<> dis(min, max);

    // Generate and return a random number within the range
    return dis(gen);
}


// Display Analytics and Errors 
void printSimulationAnalytics() { 
    printf("%s %s completed, took %f seconds, returned %s\n", SIMULATION_NAME, "Simulation", 
        totalTime, (g_TotalErrors == 0) ? "OK." : "ERROR!");

    printf("The simulation took an average of %f milliseconds per update", 
            sdkGetAverageTimerValue(&timer) / 1000.f);
    if (gpu_accel) {
        printf(" on the GPU.\n");
    } else {
        printf(" on the CPU.\n");
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Basic Linux preprocessor directive for display
    #if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
    #endif

    // Process Command Line Arguments
    processCmdLineArgs(argc, argv);

    // Print that the Simulation has started
    {
        printf("%s %s", SIMULATION_NAME, "Simulation");
        if (gpu_accel) {
            printf(" on the GPU");
        } else {
            printf(" on the CPU");
        }
        printf(" starting...\n");
    }

    // Run Simulation
    runSimulation(argc, argv);

    // Print Helpful Analytics from Simulation
    printSimulationAnalytics();

    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}



////////////////////////////////////////////////////////////////////////////////
//! Run the Interactive Particle Simulation
////////////////////////////////////////////////////////////////////////////////
bool runSimulation(int argc, char **argv)
{
    // Creates the CUTIL timer
    sdkCreateTimer(&timer);

    // Initializes GL, returning false on failure
    if (!initGL(&argc, argv)) {
        return false;
    }

    // Registers callbacks
    glutDisplayFunc(display);
    glutIdleFunc(redisplay);
    glutKeyboardFunc(handleKeyPress);
    glutKeyboardUpFunc(handleKeyRelease);
    glutSpecialFunc(handleSpecialKeyPress);
    glutSpecialUpFunc(handleSpecialKeyRelease);
    
    // Registers exit callback
    #if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
    #else
    glutCloseFunc(cleanup);
    #endif

    // Initializes Player
    p1 = new Player(0.0f, 0.0f, PLAYER_WIDTH, PLAYER_HEIGHT);

    // Creates the VBO that will store all the raw Particle data
    createParticleVBO(&particleVBO);

    // Sets up for GPU Acceleration
    if (gpu_accel) {
        setUpGPU(&particleVBO, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    }

    // Times the initialization
    sdkStartTimer(&timer);
    // Initializes Particles
    if (gpu_accel) {
        initParticlesGPU(&cuda_vbo_resource);
    } else {
        generator.seed(static_cast<unsigned int>(time(0)));
        bool particlesInitialized = initParticlesCPU();
        if (!particlesInitialized) return false;
    }
    sdkStopTimer(&timer);
    printf("Particle Initialization took %f seconds", sdkGetAverageTimerValue(&timer) / 1000.f);
    if (gpu_accel) {
        printf(" on the GPU.\n");
    } else {
        printf(" on the CPU.\n");
    }
    // Wipes timer so the initialization isn't considered in FPS computation
    sdkResetTimer(&timer);

    // Enters rendering mainloop
    glutMainLoop();

    // Returns true for success at the end
    return true;
}



////////////////////////////////////////////////////////////////////////////////
//! GL Functionality
////////////////////////////////////////////////////////////////////////////////

//  Set Up GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    // Initialize GLUT and Create the Window
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow(SIMULATION_NAME);

    // Handlers set during runSimulation

    // Initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        g_TotalErrors++;
        return false;
    }

    // Initialize a Blank (Black) Screen
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Window Dimensions are the same as Viewport Dimensions
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    SDK_CHECK_ERROR_GL();

    return true;
}


//  Create Particle VBO
////////////////////////////////////////////////////////////////////////////////
void createParticleVBO(GLuint *p_vbo)
{
    assert(p_vbo);

    // create buffer object
    glGenBuffers(1, p_vbo);

    // Bind the buffer to the target GL_ARRAY_BUFFER
    // Now all ops on GL_ARRAY_BUFFER will affect vbo (until unbound)
    glBindBuffer(GL_ARRAY_BUFFER, *p_vbo);

    // Initialize data for vbo (sets everything to NULL)
    unsigned int size = n_particles * sizeof(Particle);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    // Unbind vbo from target GL_ARRAY_BUFFER (binding to 0 unbinds the target)
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    SDK_CHECK_ERROR_GL();
}


//  Delete Particle VBO
////////////////////////////////////////////////////////////////////////////////
void deleteParticleVBO(GLuint *p_vbo)
{

    glBindBuffer(1, *p_vbo);
    glDeleteBuffers(1, p_vbo);

    *p_vbo = 0;
}



////////////////////////////////////////////////////////////////////////////////
//! Callbacks and Handlers
////////////////////////////////////////////////////////////////////////////////

//! Display Callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // Updates total and delta time (in seconds)
    {
        // Gets seconds that have elapsed since glutInit was called
        float newTotalTime = glutGet(GLUT_ELAPSED_TIME) / 1000.f;
        dt = newTotalTime - totalTime;
        totalTime = newTotalTime;
        timeOneSec += dt; // for tracking FPS
    }

    // Increases Particle Base Speed over time (simulates harder game)
    {
        // If particle staggering is on, doesn't start increasing speed until
        // all particles have spawned in (started being updated)
        int n_most_particles = MAX(MAX(n_seekers, n_cruisers), n_wanderers);
        if (!particle_stagger_start
                || n_most_particles > totalTime * PARTICLE_STAGGER_PER_SEC) {
            particle_base_speed += PARTICLE_BASE_SPEED_ACCEL * dt;
        }
    }

    if (simulation_on) {
        p1->update();

        if (gpu_accel) {
            updateParticlesGPU(&cuda_vbo_resource);
        } else {
            updateParticlesCPU();
        }
    } else {
        p1->updateOnlyColor();
    }
    
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    p1->render();

    // Particle Rendering
    {
        // Tells OpenGL we'll be rendering vertices using array instead of individually 
        // Enable vertex position and color arrays
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        // Binds the particle VBO as target for vertex arrays below
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);

        // Specifies pointers to the vertex arrays using the bound buffer object
        glVertexPointer(2, GL_FLOAT, sizeof(Particle), (void*)0);
        glColorPointer(3, GL_FLOAT, sizeof(Particle), (void*)(2 * sizeof(GLfloat)));

        // Draws the Seeker and Cruiser particles at the normal size
        glPointSize(PARTICLE_POINT_SIZE / SCALEX);
        glDrawArrays(GL_POINTS, 0, n_seekers + n_cruisers);

        // Draws the Wanderer particles at the bigger indicated size
        glPointSize(PARTICLE_POINT_SIZE * WANDERER_BIGGER_SIZE_RATIO / SCALEX);
        glDrawArrays(GL_POINTS, n_seekers + n_cruisers, n_wanderers);

        // Disables the vertex arrays
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
        
    // Modelview matrix transform not yet used in this version

    // Swaps buffers, effecitvely rendering immediately
    glutSwapBuffers();

    sdkStopTimer(&timer);

    // Calculates fps using timer and displays it in window title
    computeFPS();
}


//! Close Callback
////////////////////////////////////////////////////////////////////////////////
void cleanup()
{
    printSimulationAnalytics();

    sdkDeleteTimer(&timer);

    if (gpu_accel) {
        cleanUpGPU(cuda_vbo_resource);
    }

    if (particleVBO)
    {
        deleteParticleVBO(&particleVBO);
    }

    if (p1) {
        delete p1;
    }
}


//! Idle Handler
////////////////////////////////////////////////////////////////////////////////
// Ensures continous display by always calling the display function
void redisplay() {
    glutPostRedisplay();
}


//! Keyboard Events Handlers
////////////////////////////////////////////////////////////////////////////////
// These handlers use 2 maps to track which keys are currently being held down
// Need 4 bc GL has separate handlers for ASCII/Non-ASCII Keys and Pressing/Releasing
void handleKeyPress(unsigned char key, int x, int y) {
    keyStates[key] = true;  // Set the state of the key to pressed
    switch (key) {
        case (' ') :
            simulation_on = true;
            break;
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
    }

}
void handleKeyRelease(unsigned char key, int x, int y) {
    keyStates[key] = false; // Set the state of the key to not pressed
}
void handleSpecialKeyPress(int key, int x, int y) {
    specialKeyStates[key] = true;  // Set the state of the key to pressed
}
void handleSpecialKeyRelease(int key, int x, int y) {
    specialKeyStates[key] = false; // Set the state of the key to not pressed
}


//! Display Helper Functions
////////////////////////////////////////////////////////////////////////////////
// Computes and displays in Window Title once the simulation has started
// If the simulation has started, tells user to "Press Space to Start"
void computeFPS()
{
    char title[256];
    if (simulation_on) {
        frameCount++;
        if (timeOneSec >= 1.f)
        {
            currentFPS = static_cast<float>(frameCount) / timeOneSec;
            frameCount = 0;
            timeOneSec = 0.0f;
        }
        sprintf(title, "%s: %4.1f fps (Max 100Hz)", SIMULATION_NAME, currentFPS);
    } else {
        sprintf(title, "%s: Press Space to Start", SIMULATION_NAME);
    }
    glutSetWindowTitle(title);
}




/* Not Yet In Use 
    ////////////////////////////////////////////////////////////////////////////////
    //! Mouse event handlers
    // Not yet used for this version
    ////////////////////////////////////////////////////////////////////////////////

    //int mouse_old_x, mouse_old_y;
    //int mouse_buttons = 0;
    //float rotate_x = 0.0, rotate_y = 0.0;
    //float translate_z = -3.0;

    void mouse(int button, int state, int x, int y)
    {
        if (state == GLUT_DOWN)
        {
            mouse_buttons |= 1<<button;
        }
        else if (state == GLUT_UP)
        {
            mouse_buttons = 0;
        }

        mouse_old_x = x;
        mouse_old_y = y;
    }

    void motion(int x, int y)
    {
        float dx, dy;
        dx = (float)(x - mouse_old_x);
        dy = (float)(y - mouse_old_y);

        if (mouse_buttons & 1)
        {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        }
        else if (mouse_buttons & 4)
        {
            translate_z += dy * 0.01f;
        }

        mouse_old_x = x;
        mouse_old_y = y;
    }

    // To be used in the display function for a later version
    void transformModelView() {
        // Modelview matrix doesn't need to be changed for this version
        //glMatrixMode(GL_MODELVIEW);
        //glLoadIdentity();
        //glTranslatef(0.0, 0.0, translate_z);
        //glRotatef(rotate_x, 1.0, 0.0, 0.0);
        //glRotatef(rotate_y, 0.0, 1.0, 0.0);
    }
*/