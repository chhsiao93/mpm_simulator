import numpy as np
import math
import os
import imageio

def curve_function(x, offset=0, min_val=0.1, max_val=0.3, frequency=3):
    
    return min_val + (max_val - min_val) * (0.5 + 0.5 * np.sin(frequency * (x + offset)))
    
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