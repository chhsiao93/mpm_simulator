from sand_wheel_simulator import MPMSolver
import taichi as ti
import numpy as np
import utils
import time
n_grid = 64
dx = 1/n_grid
# initialize taichi
ti.init(arch=ti.gpu)
mpm = MPMSolver(dim=2,
                n_grid=n_grid,
                n_particle=5000,
                dt=1e-4,
                target=[0.7, 0.2])
# create environment object
global_state = np.load('scene/long_terrain_scene.npz', allow_pickle=True) # load the scene
global_state = dict(global_state)
env = utils.Environment(global_state, mpm_solver=mpm)

# action function
def action_fn(observation):
    dist_vec = observation['dist_to_target']
    return np.clip(0.1*(np.random.normal(-dist_vec[0], 0.2)), -0.03, 0.03)

# initialize GUI
gui = ti.GUI("MPM Solver - Wheel Playground", (640, 640), background_color=0x112F41, show_gui=True)
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    #### simulation ####
    # sample torque
    action = action_fn(env.observation)
    # forward simulation and get new state
    env.step(action, n_substeps=100)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # visualization
    current_wheel_center = env.observation['wheel_pos']
    x_np = env.local_state['pos']
    clr_np= env.local_state['color']
    mat_np = env.local_state['material']
    scale = 4
    gui.circles(x_np, color=clr_np, radius=1.5) # particle positions at frame s
    gui.circle(np.mean(x_np[mat_np==1],axis=0), radius=5, color=0xFF0000) # center of the wheel
    gui.circle(mpm.target, radius=5, color=0x00FFFF) # target position
    gui.show()
