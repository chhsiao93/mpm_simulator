import taichi as ti
import math
import numpy as np

@ti.data_oriented
class MPMSolver:
    # only elastic and sand materials works for now
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    material_stationary = 4
    materials = {
        'WATER': material_water,
        'ELASTIC': material_elastic,
        'SNOW': material_snow,
        'SAND': material_sand,
        'STATIONARY': material_stationary,
    }
    objs = {
        'terrain': 0,
        'sand': 1,
    }
    def __init__(self, n_particle, dim=2,
                 n_grid=64,
                 dt=1e-4,
                 target=[0.5, 0.2],
                 ):
        self.dim = dim # dimension
        self.n_particles = n_particle # number of particles counter
        # self.max_num_particles = max_num_particles # number of particles
        self.n_grid = n_grid # grid resolution for MPM
        self.dx = 1 / n_grid # grid spacing
        self.inv_dx = 1 / self.dx # inverse of grid spacing
        self.dt = dt # time step
        self.p_vol = self.dx**self.dim # particle volume
        self.p_rho = 1000 # particle density
        self.p_mass = self.p_vol * self.p_rho # particle mass
        self.gravity = 9.8
        self.target = target # target position for the center of the wheel

        # Young's modulus and Poisson's ratio
        self.E, self.nu = 1e2, 0.2
        # Lame parameters
        self.mu_0, self.lambda_0 = self.E / (2.0 * (1.0 + self.nu)), self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        # Sand parameters
        friction_angle = 45.0
        sin_phi = math.sin(math.radians(friction_angle))
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        self.x = ti.Vector.field(self.dim,
                            dtype=ti.f32,
                            shape=(self.n_particles,),
                            )
        self.v = ti.Vector.field(self.dim,
                            dtype=ti.f32,
                            shape=(self.n_particles,),
                            )
        self.grid_v_in = ti.Vector.field(dim,
                                    dtype=ti.f32,
                                    shape=(self.n_grid, self.n_grid),
                                    )
        self.grid_v_out = ti.Vector.field(self.dim,
                                    dtype=ti.f32,
                                    shape=(self.n_grid, self.n_grid),
                                    )
        self.grid_m_in = ti.field(dtype=ti.f32,
                            shape=(self.n_grid, self.n_grid),
                            )
        self.C = ti.Matrix.field(self.dim,
                            self.dim,
                            dtype=ti.f32,
                            shape=(self.n_particles,),
                            )
        self.F = ti.Matrix.field(dim,
                            dim,
                            dtype=ti.f32,
                            shape=(self.n_particles,),
                            )

        self.Jp = ti.field(dtype=ti.f32, shape=(self.n_particles,))
        self.material = ti.field(dtype=ti.i32, shape=(self.n_particles,))
        self.object = ti.field(dtype=ti.i32, shape=(self.n_particles,))
        self.color = ti.field(dtype=ti.i32, shape=(self.n_particles,))
        self.current_wheel_center = ti.Vector.field(dim, dtype=ti.f32, shape=(),)
        # An empirically optimal chunk size is 1/10 of the expected particle number
        # chunk_size = 2**20 if self.dim == 2 else 2**23
        # self.particle = ti.root.dynamic(ti.i, max_num_particles, chunk_size)
        # self.particle.place(self.x, self.v, self.C, self.F, self.Jp, self.object, self.material,
        #                         self.color)

    @ti.kernel
    def set_w(self, omega: ti.f32):
        self.compute_wheel_center()
        for p in range(self.n_particles):
            if (self.material[p] == self.material_elastic):
                dist_center = ti.math.distance(self.x[p], self.current_wheel_center[None]) # distance from center to particle
                norm_vect = ti.math.normalize(self.x[p] - self.current_wheel_center[None]) # normalized vector from center to particle
                if 0 < dist_center:#< rod_radius: # inside rod region
                    self.v[p] += (omega) * dist_center * ti.Vector([-norm_vect[1], norm_vect[0]]) # make v = omega * r

    @ti.func
    def compute_wheel_center(self):
        self.current_wheel_center[None] = [0.0, 0.0]
        count = 0
        for i in range(self.n_particles):
            if (self.material[i] == self.material_elastic):
                count+=1
                self.current_wheel_center[None] +=  self.x[i]
        self.current_wheel_center[None] /= count
    @ti.kernel
    def clear_grid(self):
        for i, j in self.grid_m_in:
            self.grid_v_in[i, j] = [0.0, 0.0]
            self.grid_m_in[i, j] = 0.0

    @ti.func
    def sand_projection(self, sigma, p):
        sigma_out = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        epsilon = ti.Vector.zero(ti.f32, self.dim) 
        for i in ti.static(range(self.dim)):
            epsilon[i] = ti.math.log(ti.math.max(ti.abs(sigma[i, i]), 1e-4))
            sigma_out[i, i] = 1.0
        tr = epsilon.sum() + self.Jp[p]
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = epsilon_hat.norm() + 1e-20
        delta_gamma = 0.0
        if tr >= 0.0:
            self.Jp[p] = tr
            
        else:
            self.Jp[p] = 0.0
                
            delta_gamma = epsilon_hat_norm + (self.dim * self.lambda_0 + 2 * self.mu_0) / (2 * self.mu_0) * tr * self.alpha #[None]
            for i in ti.static(range(self.dim)):
                sigma_out[i, i] =  ti.exp(epsilon[i]- max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])
        return sigma_out

    @ti.kernel
    def p2g(self):
        for p in range(self.n_particles):
            
            base = ti.cast(self.x[p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=2, val=1) + self.dt * self.C[p]) @ self.F[p]
            h = 1.0
            if self.material[p] == self.material_elastic:
                h = 50000.0
            mu, la = self.mu_0 * h, self.lambda_0 * h
            
            stress = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            if self.material[p] == self.material_sand:
                h = 0.1*ti.exp(10 * (1.0 - self.Jp[p])) #Hardening
                mu, la = self.mu_0 * h, self.lambda_0 * h
                U, sig, V = ti.svd(new_F) 
                
                # SIG[f, p] = sand_projection(f, sig, p)
                # F[f + 1, p] = U @ SIG[f, p] @ V.transpose()
                sig_new = self.sand_projection(sig, p)
                
                self.F[p] = U @ sig_new @ V.transpose()
                log_sig_sum = 0.0
                center = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                for i in ti.static(range(self.dim)):
                    log_sig_sum += ti.log(sig_new[i, i])
                    center[i,i] = 2.0 * mu * ti.log(sig_new[i, i]) * (1 / sig_new[i, i])
                for i in ti.static(range(self.dim)):
                    center[i,i] += la * log_sig_sum * (1 / sig_new[i, i])
                cauchy = U @ center @ V.transpose() @ self.F[p].transpose()
                        
                stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
                
            else:
                self.F[p] = new_F
                J = (new_F).determinant()
                r, s = ti.polar_decompose(new_F)
                cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                    ti.Matrix.diag(2, la * (J - 1) * J)
                stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + self.p_mass * self.C[p]
            
            
            #Loop over 3x3 grid node neighborhood
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), ti.f32) - fx) * self.dx
                    weight = w[i][0] * w[j][1]
                    self.grid_v_in[base + offset] += weight * (self.p_mass * self.v[p] +
                                                            affine @ dpos)
                    self.grid_m_in[base + offset] += weight * self.p_mass


    bound = 3

    @ti.kernel
    def grid_op(self):
        for p in range(self.n_grid * self.n_grid):
            i = p // self.n_grid
            j = p - self.n_grid * i
            inv_m = 1 / (self.grid_m_in[i, j] + 1e-10)
            v_out = inv_m * self.grid_v_in[i, j]
            v_out[1] -= self.dt * self.gravity
            if i < self.bound and v_out[0] < 0:
                v_out[0] = 0
            if i > self.n_grid - self.bound and v_out[0] > 0:
                v_out[0] = 0
            if j < self.bound and v_out[1] < 0:
                v_out[1] = 0
            if j > self.n_grid - self.bound and v_out[1] > 0:
                v_out[1] = 0
            self.grid_v_out[i, j] = v_out

    @ti.kernel
    def g2p(self):
        for p in range(self.n_particles):

            base = ti.cast(self.x[p] * self.inv_dx - 0.5, ti.i32)
            # Im = ti.rescale_index(pid, grid_m, I)
            # for D in ti.static(range(dim)):
            #     base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
            fx = self.x[p] * self.inv_dx - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            # Loop over 3x3 grid node neighborhood
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), ti.f32) - fx
                    g_v = self.grid_v_out[base[0] + i, base[1] + j]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx
            
            if self.material[p] != self.material_stationary: 
                self.v[p] = new_v
                self.x[p] = self.x[p] + self.dt * self.v[p]
                self.C[p] = new_C
    
    @ti.kernel
    def input_state(
        self,
        num_particles: ti.i32,
        pos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        material: ti.types.ndarray(),
        color: ti.types.ndarray(),
        object: ti.types.ndarray(),
        C_np: ti.types.ndarray(),
        J_np: ti.types.ndarray(),
        F_np: ti.types.ndarray(),
        ):
        for i in range(num_particles):
            x = ti.Vector.zero(ti.f32, n=self.dim)
            v = ti.Vector.zero(ti.f32, n=self.dim)
            F = ti.Matrix([[F_np[i,0,0],F_np[i,0,1]],[F_np[i,1,0],F_np[i,1,1]]], ti.f32)
            C = ti.Matrix([[C_np[i,0,0],C_np[i,0,1]],[C_np[i,1,0],C_np[i,1,1]]], ti.f32)
            x = ti.Vector([pos[i, 0], pos[i, 1]], ti.f32)
            v = ti.Vector([vel[i, 0], vel[i, 1]], ti.f32)
            J = J_np[i]
            self.seed_particle(i, x, material[i], object[i],
                               color[i], v, J, F, C)
    
    @ti.func
    def seed_particle(self, i, x, material, object, color, velocity, J, F, C):
        self.x[i] = x
        self.v[i] = velocity
        self.color[i] = color
        self.object[i] = object
        self.material[i] = material
        self.F[i] = F
        self.Jp[i] = J
        self.C[i] = C

    
    def substep(self):
        self.clear_grid()
        self.p2g()
        self.grid_op()
        self.g2p()
        
    def step(self, state, action, n_substeps=200):
        # input state s0
        self.input_state(num_particles=state['num_particles'],
                                    pos=state['pos'],
                                    vel=state['vel'],
                                    material=state['material'],
                                    color=state['color'],
                                    object=state['object'],
                                    C_np=state['C_np'],
                                    F_np=state['F_np'],
                                    J_np=state['J_np'])
        
        # foward simulation
        for _ in range(n_substeps):
            # action: apply torque to the wheel
            self.set_w(action)
            self.substep()
        # get new state s1
        return self.export_state()
    def export_state(self):
        return {
            'pos': self.x.to_numpy(),
            'vel': self.v.to_numpy(),
            'material': self.material.to_numpy(),
            'color': self.color.to_numpy(),
            'object': self.object.to_numpy(),
            'C_np': self.C.to_numpy(),
            'F_np': self.F.to_numpy(),
            'J_np': self.Jp.to_numpy(),
            'num_particles': self.n_particles,
        }