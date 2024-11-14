from typing import Optional
import numpy as np
import gymnasium as gym
from copy import deepcopy
# environment object
class WheelSandEnv(gym.Env):
    def __init__(self, global_state, mpm_solver, offset=0.0, buffer=0.03, target=[0.5, 0.2], size=6, mario=False):
        # The size of the world
        self.size = size
        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([0.5, 0.5], dtype=np.float32)
        self._target_location = np.array(target, dtype=np.float32)
        self._velocity = np.array([0.0, 0.0], dtype=np.float32)
        self._omega = np.array([0.0], dtype=np.float32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, self.size, shape=(2,), dtype=float),
                "target": gym.spaces.Box(0, self.size, shape=(2,), dtype=float),
                "velocity": gym.spaces.Box(-10, 10, shape=(2,), dtype=float),
                "omega": gym.spaces.Box(-10, 10, shape=(1,), dtype=float),
            }
        )
        # We have 4 actions, corresponding to "fast right", "slow right", "stay", "slow left", and "fast left" 
        self.action_space = gym.spaces.Discrete(5)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: -0.05,  # fast right
            1: -0.01,  # slow right
            2: 0.0,  # stay
            3: 0.01,  # slow left
            4: 0.05,  # fast left
        }
        self.initial_global_state = deepcopy(global_state)
        self.global_state = deepcopy(global_state)
        self.offset = offset
        self.buffer = buffer
        self.local_state = {}
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
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "velocity": self._velocity, "omega": self._omega}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location
            )
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Recover the world state by copy the value from the initial state
        self.global_state = deepcopy(self.initial_global_state)
        # Choose the agent's location uniformly at random
        self.offset = np.random.uniform(0.0, self.size-1.0)
        # Translate the agent/wheel location with offset
        self.global_state['pos'][self.global_state['object'] == 1] += np.array([self.offset, 0.0])
        self.initial_wheel_center = np.mean(self.global_state['pos'][self.global_state['object'] == 1], axis=0)
        self._agent_location = self.initial_wheel_center

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.abs(self._agent_location - self._target_location)[0] < 1.0:
            self._target_location = np.array([np.random.uniform(0.5, self.size-0.5), np.random.uniform(0,0.2)], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

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
        # update observation
        self._agent_location = r_com
        self._velocity = v_com
        self._omega = omega
        
    def step(self, action, n_substeps=100):
        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        self.action = self._action_to_direction[action]
        self.local_state, mask = self.find_local_state()
        if self.local_state['num_particles'] <= 0:
            print("No particles in the window")
            return self._get_obs(), 0, True, False, self._get_info()
        assert self.local_state['num_particles'] <= self.max_num_particles ,f"Number of particles in the window is {self.local_state['num_particles']}, which is larger than the maximum number of particles {self.max_num_particles}"
        self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] - self.offset # map to local coordinate
        self.local_state = self.mpm.step(self.local_state, action=self.action, n_substeps=n_substeps)
        self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] + self.offset # map back to global coordinate
        self.update_global_state(self.local_state, mask)
        self.observe(self.local_state)
        if self.mario:
            self.offset = np.clip((self._agent_location[0] - 0.5), 0, self.size-0.5) # update offset of window to follow the wheel
            

        # An environment is completed if and only if the agent has reached the target
        terminated = np.abs(self._agent_location - self._target_location)[0] < 0.02
        truncated = False
        reward = 100 if terminated else 0  # the agent is only reached at the end of the episode
        reward += -0.1 * np.linalg.norm(self._agent_location - self._target_location)  # the agent is penalized for being far from the target
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward.astype(np.float32), terminated, truncated, info 
    # def step(self, action, n_substeps=100, observation=True):
    #     self.action = action
    #     self.local_state, mask = self.find_local_state()
    #     assert self.local_state['num_particles'] > 0, "No particles in the window"
    #     assert self.local_state['num_particles'] <= self.max_num_particles ,f"Number of particles in the window is {self.local_state['num_particles']}, which is larger than the maximum number of particles {self.max_num_particles}"
    #     self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] - self.offset # map to local coordinate
    #     self.local_state = self.mpm.step(self.local_state, action=self.action, n_substeps=n_substeps)
    #     self.local_state['pos'][:, 0] = self.local_state['pos'][:, 0] + self.offset # map back to global coordinate
    #     self.update_global_state(self.local_state, mask)
    #     if observation:
    #         self.observe(self.local_state)
    #     if self.mario:
    #         self.offset = (self.observation['wheel_pos'][0] - self.initial_wheel_center[0]) # update offset of window to follow the wheel
        
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
# gym.register(
#     id="gymnasium_env/WheelSand-v0",
#     entry_point=WheelSandEnv,
# )