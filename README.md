<h1>MPM simulator</h1>

The simulator is inspired by the Taichi-Element code and uses the Taichi language to implement the Material Point Method for simulating the interaction between an elastic wheel (with grousers) and deformable terrain (sand).

<h2>Create a Scene</h2>

The scene is created in `utils.py`. The positions of the wheel's material points can be generated using the `add_spike()` function, where you can specify the wheel configuration, such as the number of grousers, center location, radius, and more. The function returns an (n_particle, dim) ndarray representing the positions of the wheel particles. Similarly, you can create a flat terrain using the `add_cube()` function. I have also defined a function, `create_wheel_scene()`, to generate the initial state of a wheel on curved terrain. You can customize your own function to create different scenes. Here's an example of how to create a scene:

```.py
import utils
state = utils.create_wheel_scene()
```

`create_wheel_scene()` will return the state of the states of particles as a dictionary, including:
1. Number of Particles
2. Position
3. Velocity
4. Material Types
5. Color
6. Object Labels (e.g., terrain or wheel)
7. C_np
8. F_np (deformation gradient)
9. J_np (plastic deformation)
<h2>Create a MPM Simulator</h2>

The mpm simulator can be created as a object:
```.py
from sand_wheel_simulator import MPMSolver
import taichi as ti
ti.init(arch=ti.gpu)
mpm = MPMSolver(dim=2,
                n_grid=64,
                n_particle=5000,
                dt=1e-4)

```
`n_particle` is a pre-defined parameter, which tells mpm solver the maximum number of particle it should handle. You can use a higher number for a high resolution scene (more particles)

<h2>Environment</h2>
Create a Environment object to handle both the scene and the mpm solver.

```.py
env = utils.Environment(state, mpm_solver=mpm)
```

<h3>States</h3>

`env` stores the whole scene in `env.global_state`. It also stores the scene which is within the simulation scope, this allows simulation in a local scope without intense memory requirment. 

<h3>Observation</h3>

`env.observation` stores the important features such as `wheel_pos`, `wheel_vel`, `wheel_omega`, `dist_to_target`

<h3>Forward Step</h2>

You can advance the simulation using `env.step()`, which takes the current state and the action (torque applied on wheel, in this case) as inputs and returns the updated state.

```.py
env.step(action, n_substeps=100)
```
`n_substeps` represents the number of steps in one forward simulation. Since the time step is very small (e.g., $dt=1×10^{−4}$), changing the action at every time step is impractical. Instead, the action is treated as a constant torque applied to the wheel at each substep.

<h2>Run Example</h2>
<h3>Simple case</h3>
You can run an example where the wheel's actions are sampled from a normal distribution, with the magnitude of the action being proportional to the distance to the target in the x-direction.
```.py
python run_simulator.py
```
![simple](https://github.com/user-attachments/assets/b7b80e2f-5a04-4e8d-869c-c2178b69b3fc)

<h3>Mario mode</h3>
The simulation scene may be large, requiring significant memory to simulate the entire scene. However, most of the sand particles remain static since there is no dynamic actor (e.g., a wheel) present. To save memory, we use a moving window that simulates only the particles near the wheel. In each iteration, the simulator extracts the local state from the global state to input into the MPM simulator. After each simulation step, the global state is updated. To enable this feature, set `mario=True` when initializing `env`:

```.py
env = utils.Environment(state, mpm_solver=mpm, mario=True)
```

The following animation shows the simulation scope follows the wheel location.

![mario_local](https://github.com/user-attachments/assets/a2a626da-79be-4e09-99d8-b2ae2b05b8cd)|

You can run this example by running:

`python run_mario.py` 

<h4>Auxiliary Simulation Window</h4>

Main Simulation Window            |  Main + Aux
:-------------------------:|:-------------------------:
![mario_global](https://github.com/user-attachments/assets/36b1d035-7927-4169-b86e-63fa5c247fac)|![mario_aux](https://github.com/user-attachments/assets/97c449d7-7026-4452-8cf2-94b0ff4709c1)
