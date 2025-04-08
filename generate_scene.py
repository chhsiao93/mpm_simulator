import utils
import numpy as np

def create_long_terrain_scene(dim = 2, dx=1/64, density_scale=1):
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    material_stationary = 4
    objs = {
        'wheel': 1, # always 1
        'sand0': 45,
        'sand1': 70,
        'sand2': 10,
    }
    # Create initial state of wheel and sand
    mat = np.array([]) # material list
    clr = np.array([]) # color list
    obj = np.array([]) # object list
    cube = utils.add_cube(lower_corner=[0.01, 0.01], 
                            cube_size=[6.0, 0.25], 
                            dx=dx, sample_density=(2**dim)*density_scale, 
                            dim=dim)

    is_valid = (cube[:,1] < utils.curve_function(cube[:,0]))
    cube = cube[is_valid]
    mat = np.append(mat, np.ones(cube.shape[0]) * material_sand)
    bd1 = 1.0
    bd2 = 2.0
    is_sand0 = (cube[:,0] < bd1) # x < 2.0
    is_sand1 = (cube[:,0] >= bd1) & (cube[:,0] < bd2)
    is_sand2 = (cube[:,0] >= bd2) # x >= 3.0
    
    clr = np.append(clr, np.ones(cube.shape[0]) * 0xFFFFFF) # sand0 is white, sand1 is yellow, sand2 is red
    clr[is_sand0] = 0xFFFFFF # white
    clr[is_sand1] = 0x808080 # gray
    clr[is_sand2] = 0x8B4513 # brown
    obj = np.append(obj, np.zeros(cube.shape[0]))
    obj[is_sand0] = objs['sand0']
    obj[is_sand1] = objs['sand1']
    obj[is_sand2] = objs['sand2']
    
    ### add spikes/wheel and get particle positions for spikes
    fan_center = [0.5, 0.5] # center of the wheel/fan
    rod_radius = 0.1 # length of the grouser of the wheel
    wheel = utils.add_spikes(sides=8,
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
    
    assert scene['object'].shape[0] == scene['num_particles'], "object and num_particles mismatch"
    return scene


if __name__ == '__main__':
    scene = create_long_terrain_scene(density_scale=1.0)
    # save the scene dictionary
    np.savez('scene/color_terrain_scene.npz', **scene)