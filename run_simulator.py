from sand_wheel_simulator import MPMSolver
import taichi as ti
import numpy as np
import utils
import time
n_grid = 64
dx = 1/n_grid
state = utils.create_wheel_scene(density_scale=1, dim=2, dx=dx)

ti.init(arch=ti.gpu)
mpm = MPMSolver(dim=2,
                n_grid=n_grid,
                n_particle=state['num_particles'],
                dt=1e-4,
                target=[0.5, 0.2])
gui = ti.GUI("MPM Solver - Wheel Playground", (640, 640), background_color=0x112F41, show_gui=True)
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    dist = mpm.target[0] - mpm.current_wheel_center[None][0] # distance to target
    action = 0.1*(np.random.normal(-dist, 0.2)) # random action
    start_time = time.time()
    state = mpm.step(state, action=action, n_substeps=100) # forward simulation and get new state
    end_time = time.time()
    print(f"Step time: {end_time - start_time:.6f} seconds")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # visualization
    current_wheel_center_np = mpm.current_wheel_center.to_numpy()
    img_count = 0
    x_np = mpm.x.to_numpy()
    clr_np= mpm.color.to_numpy()
    mat_np = mpm.material.to_numpy()
    scale = 4
    gui.circles(x_np, color=clr_np, radius=1.5) # particle positions at frame s
    gui.circle(np.mean(x_np[mat_np==1],axis=0), radius=5, color=0xFF0000) # center of the wheel
    gui.circle(mpm.target, radius=5, color=0x00FFFF) # target position
    gui.show()
