from sand_wheel_simulator import MPMSolver
import taichi as ti
import numpy as np
import utils
import time
target1 = [2.5,0.2]
target2 = [1.5,0.2]
# initialize taichi
ti.init(arch=ti.gpu)
mpm = MPMSolver(dim=2,
            n_grid=64,
            n_particle=5000,
            dt=1e-4,
            target=target1)
# create environment object
global_state = np.load('scene/long_terrain_scene.npz', allow_pickle=True) # load the scene
global_state = dict(global_state)
offset = 0.0
env = utils.Environment(global_state, mpm_solver=mpm, offset=offset, buffer=0.03, mario=True)

# initialize GUI
gui = ti.GUI("MPM Solver - Wheel Playground", (640, 640), background_color=0x112F41, show_gui=True)

# action function
def action_fn(observation):
    dist_vec = observation['dist_to_target']
    return np.clip(0.1*(np.random.normal(-dist_vec[0], 0.2)), -0.03, 0.03)

for frame in range(300):
    #### simulation ####
    # sample torque
    action = action_fn(env.observation)
    # forward simulation and get new state
    env.step(action, n_substeps=100)
    # additional window outside the observation window
    env.adjust_step(n_substeps=100)
    
    
    
    
    #### add condition to change the target position ####
    if (env.target[0]==target1[0]) & (env.observation['wheel_pos'][0] > target1[0]):
        env.target = target2 # change target position
        print(f'Target position changed to {env.target}')
    elif (env.target[0]==target2[0]) & (env.observation['wheel_pos'][0] < target2[0]):
        env.target = target1 # change target position
        print(f'Target position changed to {env.target}')
                                         
                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    #### visualization ####
    scale = np.array([6,6])
    x_np = env.global_state['pos']/scale
    clr_np= env.global_state['color']
    mat_np = env.global_state['material']
    wheel_center = np.mean(x_np[env.global_state['object'] == 1], axis=0)
    gui.circles(x_np, color=clr_np, radius=1.5) # particle positions at frame s
    gui.circle(wheel_center, radius=5, color=0xFF0000) # center of the wheel
    gui.circle(env.target/scale, radius=5, color=0xde1a24) # target position
    if env.offset > 1.0:
        gui.rect(topleft=[env.offset-1,0]/scale, bottomright=[env.offset,1]/scale, color=0x79d003) # left auxiliary window
    if env.offset < 5.0:
        gui.rect(topleft=[env.offset+1,0]/scale, bottomright=[env.offset+2,1]/scale, color=0x79d003) # right auxiliary window
    gui.rect(topleft=[env.offset,0]/scale, bottomright=[env.offset+1,1]/scale, color=0xf89f0c) # observation window
    
    # show the observation
    aligner = 0.01
    gui.text(content=f"Torque (Action): {action:.2f}", pos=(aligner, 0.95), font_size=16, color=0xFFFFFF)  
    gui.text(content=f"Distance to target: {env.observation['dist_to_target'][0]:.2f}", pos=(aligner, 0.92), font_size=16, color=0xFFFFFF)
    gui.text(content=f"Wheel Pos: ({env.observation['wheel_pos'][0]:.2f}, {env.observation['wheel_pos'][1]:.2f})", pos=(aligner, 0.89), font_size=16, color=0xFFFFFF)
    gui.text(content=f"Wheel Vel: ({env.observation['wheel_vel'][0]:.2f}, {env.observation['wheel_vel'][1]:.2f})", pos=(aligner, 0.86), font_size=16, color=0xFFFFFF)
    gui.text(content=f"Wheel Omega: {env.observation['wheel_omega']:.2f}", pos=(aligner, 0.83), font_size=16, color=0xFFFFFF)
    offset = env.offset # update offset
    gui.show()
#     gui.show(f'output/frame_{frame:04d}.png')
    
# utils.png_to_gif('output', 'output/mario_global.gif', fps=10)
