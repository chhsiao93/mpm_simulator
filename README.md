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
                n_particle=state['num_particles'],
                dt=1e-4)

```
<h2>Forward Step</h2>

You can advance the simulation using `mpm.step()`, which takes the current state and the action (torque applied on wheel, in this case) as inputs and returns the updated state.

```.py
new_state = mpm.step(old_state, action, n_substeps=100)
```
`n_substeps` represents the number of steps in one forward simulation. Since the time step is very small (e.g., $dt=1×10^{−4}$), changing the action at every time step is impractical. Instead, the action is treated as a constant torque applied to the wheel at each substep.
<h2>Run Example</h2>

You can run an example where the wheel's actions are sampled from a normal distribution, with the magnitude of the action being proportional to the distance to the target in the x-direction.
```.py
python run_simulator.py
```
