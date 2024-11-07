import numpy as np
import math
import os
import imageio

def curve_function(x, offset=0, min_val=0.15, max_val=0.3, frequency=3):
    # combination of two sine waves
    
    # return min_val + (max_val - min_val) * (0.5 + 0.5 * np.sin(frequency * (x + offset)) + 0.5 * np.sin(2*frequency * (x + offset)))
    sine_wave1 = np.sin(frequency * (x + offset))
    sine_wave2 = np.sin(2 * frequency * (x + offset))
    combined_wave = 0.5 + 0.5 * sine_wave1 + 0.5 * sine_wave2
    return min_val + (max_val - min_val) * combined_wave
    
def add_curve_terrain(dx, sample_density=None, min_val=0.2, max_val=0.4, dim=2):

    if sample_density is None:
        sample_density = 2**dim
    num_new_particles = int(sample_density * (1/dx)**dim + 1)
    xp = np.random.random((num_new_particles, dim))*[1.0,max_val]
    is_valid = (xp[:,1] < curve_function(xp[:,0])) & (xp[:,1] > 0.02) & (xp[:,0] < 0.98) & (xp[:,0] > 0.02)
    terrain = xp[is_valid]
    return terrain


def add_cube(lower_corner,
                cube_size,
                dx,
                sample_density=None,
                dim = 2):
    if sample_density is None:
        sample_density = 2**dim
    vol = 1
    for i in range(dim):
        vol = vol * cube_size[i]
    num_new_particles = int(sample_density * vol / dx**dim + 1)

    xp = np.random.random((num_new_particles, dim))* cube_size + lower_corner
    return xp
### add spikes - start
def add_spikes(
    sides,
    center,
    radius,
    width,
    dx,
    dim=2,
    sample_density=None,
    
):
    inv_dx = 1.0 / dx
    if dim != 2:
        raise ValueError("Add Spikes only works for 2D simulations")
    if sample_density is None:
        sample_density = 2**dim
    dist_side = (width/2)/(np.tan(math.pi/sides)) # center to side
    dist_vertice = (width/2)/(np.sin(math.pi/sides)) # center to vertice
    area_ngon = 0.5 * (dist_vertice * inv_dx)**2 * np.sin(
        2 * math.pi / sides) * sides # center Ngon
    area_blade = width * (radius - dist_side) * inv_dx**2 * sides # and spikes
    
    num_particles = int(math.ceil((area_ngon + area_blade) * sample_density))

    # xp = seed_spike(num_particles, sides, radius, width, material, color)
    xp = random_point_in_unit_spike(sides, radius, width, num_particles) * [radius, radius] + center

    return xp

def random_point_in_unit_spike(sides, radius, width, num_particles=1):
    pts = np.zeros((num_particles, 2))
    central_angle = 2 * math.pi / sides
    for p in range(num_particles):
        while True:
            isin = False
            point = np.random.random(2) * 2 - 1 #-1 to 1
            for i in range(sides):
                p_B = np.array([0,0])
                p_C = np.array([1,1]) * np.array([np.cos(central_angle*i),np.sin(central_angle*i)])
                p_A = np.array([width/2/radius,width/2/radius]) * np.array([np.cos(central_angle*i+math.pi/2),np.sin(central_angle*i+math.pi/2)])
                # check if the point is in the rectangle (half of blade)
                if (np.dot(p_B-p_A, point-p_A) >= 0) & (np.dot(p_B-p_C, point-p_C) >= 0) & (np.dot(p_A-p_B, point-p_B) >= 0) & (np.dot(p_C-p_B, point-p_B) >= 0):
                    isin = True
                    break
                # check if the point is in the rectangle (the other half of blade)
                elif (np.dot(p_B+p_A, point+p_A) >= 0) & (np.dot(p_B-p_C, point-p_C) >= 0) & (np.dot(-p_A-p_B, point-p_B) >= 0) & (np.dot(p_C-p_B, point-p_B) >= 0):
                    isin = True
                    break
            if isin:
                break    
        pts[p] = point
    return pts

### add spikes - end

def create_long_terrain_scene(dim = 2, dx=1/64, density_scale=1):
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    material_stationary = 4
    objs = {
        'terrain': 0,
        'wheel': 1,
    }
    # Create initial state of wheel and sand
    mat = np.array([]) # material list
    clr = np.array([]) # color list
    obj = np.array([]) # object list
    cube = add_cube(lower_corner=[0.5, 0.5], 
                            cube_size=[0.1, 0.1], 
                            dx=dx, sample_density=2*dim**density_scale, 
                            dim=dim)
    mat = np.append(mat, np.ones(cube.shape[0]) * material_elastic)
    clr = np.append(clr, np.ones(cube.shape[0]) * 0xFFFFFF)
    obj = np.append(obj, np.ones(cube.shape[0]) * objs['wheel'])
    xps = cube
    scene = {}
    scene['num_particles'] = xps.shape[0]
    scene['pos'] = xps.astype(np.float32)
    scene['vel'] = np.zeros_like(xps).astype(np.float32) + np.array([1.0, 0.5]).astype(np.float32)
    scene['material'] = mat.astype(np.int32)
    scene['color'] = clr.astype(np.int32)
    scene['object'] = obj.astype(np.int32)
    scene['C_np'] = np.zeros((scene['num_particles'], 2, 2)).astype(np.float32)
    scene['F_np'] = np.tile(np.eye(2), (scene['num_particles'], 1, 1)).astype(np.float32)
    scene['J_np'] = (np.ones_like(mat) * (mat!=material_sand)).astype(np.float32)
    return scene
def create_wheel_scene(dim = 2, dx=1/64, density_scale=1):
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    material_stationary = 4
    objs = {
        'terrain': 0,
        'wheel': 1,
    }
    # Create initial state of wheel and sand
    mat = np.array([]) # material list
    clr = np.array([]) # color list
    obj = np.array([]) # object list
    ### add a cube and get particle positions for a cube
    # cube = add_cube(lower_corner=[0.01, 0.01], 
    #                         cube_size=[0.95, 0.15], 
    #                         dx=dx, sample_density=2*dim**density_scale, 
    #                         dim=dim)
    cube = add_curve_terrain(dx=dx, sample_density=2*dim**density_scale, min_val=0.2, max_val=0.4, dim=dim)
    mat = np.append(mat, np.ones(cube.shape[0]) * material_sand)
    clr = np.append(clr, np.ones(cube.shape[0]) * 0xFFFFFF)
    obj = np.append(obj, np.ones(cube.shape[0]) * objs['terrain'])

    ### add spikes/wheel and get particle positions for spikes
    fan_center = [0.2, 0.4] # center of the wheel/fan
    rod_radius = 0.1 # length of the grouser of the wheel
    wheel = add_spikes(sides=8,
                            center=fan_center,
                            radius=rod_radius,
                            width=0.02, dx=dx,
                            dim=dim,
                            sample_density=4*dim**density_scale)
    mat = np.append(mat, np.ones(wheel.shape[0]) * material_elastic)
    clr = np.append(clr, np.ones(wheel.shape[0]) * 0xFFAAAA)
    obj = np.append(obj, np.ones(wheel.shape[0]) * objs['wheel'])
    xps = np.concatenate([cube, wheel], axis=0)
    scene = {}
    scene['num_particles'] = xps.shape[0]
    scene['pos'] = xps.astype(np.float32)
    scene['vel'] = np.zeros_like(xps).astype(np.float32)
    scene['material'] = mat.astype(np.int32)
    scene['color'] = clr.astype(np.int32)
    scene['object'] = obj.astype(np.int32)
    scene['C_np'] = np.zeros((scene['num_particles'], 2, 2)).astype(np.float32)
    scene['F_np'] = np.tile(np.eye(2), (scene['num_particles'], 1, 1)).astype(np.float32)
    scene['J_np'] = (np.ones_like(mat) * (mat!=material_sand)).astype(np.float32)
    return scene

def create_collide_scene(dim = 2, dx=1/64, density_scale=1):
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    material_stationary = 4
    objs = {
        'cube1': 0,
        'cube2': 1,
    }
    # Create initial state of wheel and sand
    mat = np.array([]) # material list
    clr = np.array([]) # color list
    obj = np.array([]) # object list
    ### add a cube and get particle positions for a cube
    cube1 = add_cube(lower_corner=[0.2, 0.2], 
                            cube_size=[0.2, 0.2], 
                            dx=dx, sample_density=density_scale*2**dim, 
                            dim=dim)
    
    # cube = add_curve_terrain(dx=dx, sample_density=2*dim**density_scale, min_val=0.2, max_val=0.4, dim=dim)
    mat = np.append(mat, np.ones(cube1.shape[0]) * material_sand)
    clr = np.append(clr, np.ones(cube1.shape[0]) * 0xFFFFFF)
    obj = np.append(obj, np.ones(cube1.shape[0]) * objs['cube1'])

    cube2 = add_cube(lower_corner=[0.6, 0.22], 
                            cube_size=[0.3, 0.25], 
                            dx=dx, sample_density=density_scale*2**dim, 
                            dim=dim)
    mat = np.append(mat, np.ones(cube2.shape[0]) * material_sand)
    clr = np.append(clr, np.ones(cube2.shape[0]) * 0xFFFFF0)
    obj = np.append(obj, np.ones(cube2.shape[0]) * objs['cube2'])
    xps = np.concatenate([cube1, cube2], axis=0)
    scene = {}
    scene['num_particles'] = xps.shape[0]
    scene['pos'] = xps.astype(np.float32)
    v0 = np.zeros_like(xps).astype(np.float32)
    v0[obj==objs['cube1']] = np.array([2.3, 0.2])
    v0[obj==objs['cube2']] = np.array([-2.0, 0.2])
    scene['vel'] =  v0
    scene['material'] = mat.astype(np.int32)
    scene['color'] = clr.astype(np.int32)
    scene['object'] = obj.astype(np.int32)
    scene['C_np'] = np.zeros((scene['num_particles'], 2, 2)).astype(np.float32)
    scene['F_np'] = np.tile(np.eye(2), (scene['num_particles'], 1, 1)).astype(np.float32)
    scene['J_np'] = (np.ones_like(mat) * (mat!=material_sand)).astype(np.float32)
    return scene

def png_to_gif(png_dir, output_file, fps):
    images = []
    png_files = sorted((os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith('.png')))
    for filename in png_files:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_file, images, duration=1000*1/fps)

# environment object
class Environment():
    def __init__(self, global_state, mpm_solver, offset=0.0, buffer=0.03, target=[0.5, 0.2], mario=False):
        self.global_state = global_state
        self.offset = offset
        self.buffer = buffer
        self.local_state = {}
        self.observation = {'wheel_pos': None, 
                            'wheel_vel': None, 
                            'wheel_omega': None,
                            'dist_to_target': None}
        self.initial_wheel_center = np.mean(self.global_state['pos'][self.global_state['object'] == 1], axis=0)
        self.action = 0.0
        self.mpm = mpm_solver
        self.target = self.mpm.target
        self.sim_size = 1.0 # currently only support [0-1] simualtion size
        self.max_num_particles = self.mpm.max_num_particles
        self.mario = mario # simulation window follows the wheel center if True
        # initialize observation
        self.local_state, _ = self.find_local_state()
        self.observe(self.local_state)

    
    def find_local_state(self):
        local_state = {}
        # find particles in the window
        mask = (self.global_state['pos'][:, 0] > self.offset+self.buffer) & (self.global_state['pos'][:, 0] < self.offset+self.sim_size-self.buffer) & (self.global_state['pos'][:, 1] > self.buffer) & (self.global_state['pos'][:, 1] < self.sim_size-self.buffer)
        local_state['num_particles'] = mask.sum()
        local_state['pos'] = self.global_state['pos'][mask]
        local_state['vel'] = self.global_state['vel'][mask]
        local_state['material'] = self.global_state['material'][mask]
        local_state['color'] = self.global_state['color'][mask]
        local_state['object'] = self.global_state['object'][mask]
        local_state['C_np'] = self.global_state['C_np'][mask]
        local_state['F_np'] = self.global_state['F_np'][mask]
        local_state['J_np'] = self.global_state['J_np'][mask]
        return local_state, mask
    
    def update_global_state(self, local_state, mask):
        local_num_particles = local_state['num_particles']
        assert mask.sum() == local_num_particles
        self.global_state['pos'][mask] = local_state['pos'][:local_num_particles]
        self.global_state['vel'][mask] = local_state['vel'][:local_num_particles]
        self.global_state['material'][mask] = local_state['material'][:local_num_particles]
        self.global_state['color'][mask] = local_state['color'][:local_num_particles]
        self.global_state['object'][mask] = local_state['object'][:local_num_particles]
        self.global_state['C_np'][mask] = local_state['C_np'][:local_num_particles]
        self.global_state['F_np'][mask] = local_state['F_np'][:local_num_particles]
        self.global_state['J_np'][mask] = local_state['J_np'][:local_num_particles]
        
    def compute_com_velocity(self, pos, vel):
        # Calculate the center of mass position and velocity in 2D
        r_com = np.mean(pos, axis=0)
        v_com = np.mean(vel, axis=0)
        return r_com, v_com

    def compute_omega(self, pos, vel, r_com, v_com):
        # Relative position and velocity with respect to the COM in 2D
        r_rel = pos - r_com  # Shape: (N, 2)
        v_rel = vel - v_com  # Shape: (N, 2)
        
        # Compute the perpendicular component of the cross product (z-component in 3D)
        omega_numerator = np.sum(r_rel[:, 0] * v_rel[:, 1] - r_rel[:, 1] * v_rel[:, 0])
        
        # Compute the sum of squared magnitudes of r_rel for the denominator
        omega_denominator = np.sum(r_rel[:, 0]**2 + r_rel[:, 1]**2)
        
        # Calculate omega (scalar angular velocity around the z-axis)
        omega = omega_numerator / omega_denominator if omega_denominator != 0 else 0.0
        
        return omega
    
    def observe(self, local_state):
        # compute current wheel omega, velocity, position
        wheel_particle_pos = local_state['pos'][local_state['object'] == 1]
        wheel_particle_vel = local_state['vel'][local_state['object'] == 1]
        r_com, v_com = self.compute_com_velocity(wheel_particle_pos, wheel_particle_vel)
        omega = self.compute_omega(wheel_particle_pos, wheel_particle_vel, r_com, v_com)
        # compute distance between target and current wheel center
        dist_to_target = self.target - r_com
        self.observation['dist_to_target'] = dist_to_target
        self.observation['wheel_pos'] = r_com
        self.observation['wheel_vel'] = v_com
        self.observation['wheel_omega'] = omega
        
    def step(self, action, n_substeps=100, observation=True):
        self.action = action
        self.local_state, mask = self.find_local_state()
        assert self.local_state['num_particles'] > 0, "No particles in the window"
        assert self.local_state['num_particles'] <= self.max_num_particles ,f"Number of particles in the window is {self.local_state['num_particles']}, which is larger than the maximum number of particles {self.max_num_particles}"
        self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] - self.offset # map to local coordinate
        self.local_state = self.mpm.step(self.local_state, action=self.action, n_substeps=n_substeps)
        self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] + self.offset # map back to global coordinate
        self.update_global_state(self.local_state, mask)
        if observation:
            self.observe(self.local_state)
        if self.mario:
            self.offset = (self.observation['wheel_pos'][0] - self.initial_wheel_center[0]) # update offset of window to follow the wheel
        
    def adjust_step(self, n_substeps=100):
        # fix left side of the observation window
        if self.offset > 1.0: 
            self.offset -= 1.0
            self.local_state, mask = self.find_local_state()
            assert self.local_state['num_particles'] > 0, "No particles in the window"
            self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] - self.offset # map to local coordinate
            self.local_state = self.mpm.step(self.local_state, action=0.0, n_substeps=n_substeps)
            self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] + self.offset # map back to global coordinate
            self.update_global_state(self.local_state, mask)
            self.offset += 1.0
        # fix right side of the observation window
        if self.offset < 5.0: 
            self.offset += 1.0
            self.local_state, mask = self.find_local_state()
            assert self.local_state['num_particles'] > 0, "No particles in the window"
            self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] - self.offset # map to local coordinate
            self.local_state = self.mpm.step(self.local_state, action=0.0, n_substeps=n_substeps)
            self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] + self.offset # map back to global coordinate
            self.update_global_state(self.local_state, mask)
            self.offset -= 1.0
        self.local_state, mask = self.find_local_state() # recompute local state