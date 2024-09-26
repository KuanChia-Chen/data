import copy
import numpy as np
import os
import time
from env.genericenv import GenericEnv
from util.colors import FAIL, WARNING, ENDC
import mediapipe as mp
from util.quaternion import euler2so3

class JayStandHandEnv(GenericEnv):
    """This is the no-clock locomotion env. It implements the bare minimum for locomotion, such as
    velocity commands. More complex no-clock locomotion envs can inherit from this class
    """
    def __init__(
        self,
        robot_name: str,
        reward_name: str,
        simulator_type: str,
        terrain: str,
        policy_rate: int,
        dynamics_randomization: bool,
        state_noise: float,
        state_est: bool,
        integral_action: bool = False
    ):
        super().__init__(
            robot_name=robot_name,
            reward_name=reward_name,
            simulator_type=simulator_type,
            terrain=terrain,
            policy_rate=policy_rate,
            dynamics_randomization=dynamics_randomization,
            state_noise=state_noise,
            state_est=state_est,
            integral_action=integral_action
        )

        # Command randomization ranges
        if robot_name == "digit":
            self.height_bounds = [0.5, 1.25]
            self.r_arm_x_bounds = [-0.45, 0.6]
            self.r_arm_y_bounds = [0.2, -0.8]
            self.r_arm_z_bounds = [-0.2, 1.4]
            self.l_arm_x_bounds = [-0.45, 0.6]
            self.l_arm_y_bounds = [-0.2, 0.8]
            self.l_arm_z_bounds = [-0.2, 1.4]
            self.reset_states = np.load(os.path.dirname(os.path.realpath(__file__)) + "/digit_init_data.npz")
        else:
            raise ValueError(f"{FAIL}Unknown robot name: {robot_name}{ENDC}")
        
        self.num_reset = self.reset_states["pos"].shape[0]

        self._randomize_commands_bounds = [100, 200] # in episode length

        self.cmd_height = 0.8
        self.cmd_r_arm_x = 0.1
        self.cmd_r_arm_y = -0.25
        self.cmd_r_arm_z = 0.0
        self.cmd_l_arm_x = 0.1
        self.cmd_l_arm_y = 0.25
        self.cmd_l_arm_z = 0.0
        self.base_adr = self.sim.get_body_adr(self.sim.base_body_name)
        self.command_counter = 0
        self.input_dir = './output_frames'
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.image_count = 0
        self.backup_movement = [0.15, 0.2, 0.3, 0.15, 0.2, 0.3, 0.8]
        self.push_marker = None
        self.test_value = 0
        # self.dummy_count = 24
        self.hand_mimic_bit = 1

        # Only check obs if this envs is inited, not when it is parent:
        if self.__class__.__name__ == "LocomotionEnv" and self.simulator_type != "ar_async":
            self.check_observation_action_size()

    @property
    def observation_size(self):
        return super().observation_size + 8 #+ self.dummy_count # height and r_shoulder command

    @property
    def extra_input_names(self):
        cmd_list = []
        # cmd_list += ['hand_mimic_bit']
        cmd_list += ['cmd-height', 'cmd_r_arm_x', 'cmd_r_arm_y', 'cmd_r_arm_z', 'cmd_l_arm_x', 'cmd_l_arm_y', 'cmd_l_arm_z']

        # Add 27 empty elements (None or "")
        # cmd_list += [0] * 24
        return cmd_list

    def reset(self, interactive_evaluation=False):
        self.randomize_commands_at = np.random.randint(*self._randomize_commands_bounds)
        self.randomize_commands()

        self.push_force = np.random.uniform(0, 30, size = 2)
        self.push_duration = np.random.randint(5, 10)
        self.push_start_time = np.random.uniform(100, 200)

        self.reset_simulation()
        rand_ind = np.random.randint(self.num_reset)
        reset_qpos = copy.deepcopy(self.reset_states["pos"][rand_ind, :])
        reset_qpos[0:2] = np.zeros(2)
        self.sim.reset(qpos = reset_qpos, qvel = self.reset_states["vel"][rand_ind, :])

        self.interactive_evaluation = interactive_evaluation
        if interactive_evaluation:
            self._update_control_commands_dict()

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        self.max_foot_vel = 0

        return self.get_state()

    def step(self, action: np.ndarray, movement: np.ndarray):
        #print("collision = ",self.sim.knee_walking_list)
        #base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        # r_hand_pose = self.sim.get_site_pose(self.sim.hand_site_name[1])
        # print("body base position = ", self.sim.get_body_pose(self.sim.base_body_name))
        # print("right hand position = ", self.sim.get_site_pose(self.sim.hand_site_name[1]))
        # print("left hand position = ", self.sim.get_site_pose(self.sim.hand_site_name[0]))
        # print("relative pose = ", self.sim.get_relative_pose(base_pose, self.sim.get_site_pose(self.sim.hand_site_name[1])))
        # print("name = ",self.sim.base_body_name)

        if movement is not None:
            
            if len(movement) == 0:
                movement = self.backup_movement
            
            movement = np.array(movement, dtype=float)

            print("movement = ", movement)
            self.backup_movement = movement
            base_pose = self.sim.get_body_pose(self.sim.base_body_name)
            
            self.cmd_height = movement[6]
            self.cmd_r_arm_x = base_pose[0] + movement[5]
            self.cmd_r_arm_y = base_pose[1] - movement[4]
            self.cmd_r_arm_z = movement[3]
            self.cmd_l_arm_x = base_pose[0] + movement[2]
            self.cmd_l_arm_y = base_pose[1] + movement[1]
            self.cmd_l_arm_z = movement[0]

            self._update_control_commands_dict()

        # print("self.cmd_r_arm_x = ",self.cmd_r_arm_x)
        # print("self.cmd_r_arm_y = ",self.cmd_r_arm_y)
        # print("self.cmd_r_arm_z = ",self.cmd_r_arm_z)
        # print("self.cmd_l_arm_x = ",self.cmd_l_arm_x)
        # print("self.cmd_l_arm_y = ",self.cmd_l_arm_y)
        # print("self.cmd_l_arm_z = ",self.cmd_l_arm_z)
        
        self.policy_rate = self.default_policy_rate
        if self.dynamics_randomization:
            self.policy_rate += np.random.randint(0, 6)

        # # Step simulation by n steps. This call will update self.tracker_fn.
        # simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        # self.step_simulation(action, simulator_repeat_steps, integral_action=self.integral_action)

        # # Reward for taking current action before changing quantities for new state
        # self.compute_reward(action)

        # # self.add_marker()

        # #self.apply_random_forces()

        # self.traj_idx += 1
        # self.last_action = action

        # if self.traj_idx % self.randomize_commands_at == 0 and not self.interactive_evaluation:
        #     self.randomize_commands()

        # if not self.interactive_evaluation:
        #     if self.push_start_time <= self.traj_idx < self.push_start_time + self.push_duration:
        #         self.sim.data.xfrc_applied[self.base_adr, 0:2] = self.push_force
        #     elif self.traj_idx == self.push_start_time + self.push_duration:
        #         self.sim.data.xfrc_applied[self.base_adr, 0:2] = np.zeros(2)


        # return self.get_state(), self.reward, self.compute_done(), {'rewards': self.reward_dict}
    def hw_step(self):
        pass

    def _get_state(self):
        # dummy_state = np.zeros(self.dummy_count)
        state = np.concatenate((
            self.get_robot_state(),
            # self.hand_mimic_bit, 
            [self.cmd_height, self.cmd_r_arm_x, self.cmd_r_arm_y, self.cmd_r_arm_z, self.cmd_l_arm_x, self.cmd_l_arm_y, self.cmd_l_arm_z]#,dummy_state
        ))
        return state

    def add_marker(self):

        base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.045, 0.045, 0.045]
        color = [1, 0, 0]
        rgba = np.concatenate((color, np.ones(1)))
        pos = [base_pose[0], base_pose[1], base_pose[2]]
        marker_params = ["sphere", "", pos, size, rgba, so3]

        if self.push_marker is None:
            self.push_marker = self.sim.viewer.add_marker(*marker_params)

        self.sim.viewer.update_marker_position(self.push_marker, pos)

    def randomize_commands(self):
        
        r_cmd_x, r_cmd_y, r_cmd_z, l_cmd_x, l_cmd_y, l_cmd_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        self.cmd_height = np.random.uniform(*self.height_bounds)

        while 1:
            r_cmd_x = np.random.uniform(*self.r_arm_x_bounds)
            r_cmd_y = np.random.uniform(*self.r_arm_y_bounds)
            r_cmd_z = np.random.uniform(*self.r_arm_z_bounds)
            l_cmd_x = np.random.uniform(*self.l_arm_x_bounds)
            l_cmd_y = np.random.uniform(*self.l_arm_y_bounds)
            l_cmd_z = np.random.uniform(*self.l_arm_z_bounds)


            if not (r_cmd_x > -0.0975 and r_cmd_x < 0.0975 and r_cmd_y < 0.087 and r_cmd_y > -0.087 and r_cmd_z < 0.49 and r_cmd_z > -0.1):
                if not (l_cmd_x > -0.0975 and l_cmd_x < 0.0975 and l_cmd_y < 0.087 and r_cmd_y > -0.087 and l_cmd_z < 0.49 and l_cmd_z > -0.1):
                    break

        self.cmd_r_arm_x = base_pose[0] + r_cmd_x
        self.cmd_r_arm_y = base_pose[1] + r_cmd_y
        self.cmd_r_arm_z = r_cmd_z
        self.cmd_l_arm_x = base_pose[0] + l_cmd_x
        self.cmd_l_arm_y = base_pose[1] + l_cmd_y
        self.cmd_l_arm_z = l_cmd_z

    # def apply_random_forces(self):

    #     self.push_force = [2,2,2]#
    #     self.sim.data.xfrc_applied[self.base_adr, 0:3] = self.push_force

    #     rand_force_prob = 1/150 # twice in a 300 ep window
    #     # rand_torque_prob = 1/150 if self.zero_cmd else 1/300
    #     max_force = 150
    #     # if np.random.random() < rand_force_prob:
    #     self.force_window = np.random.randint(10, 25)
    #     sign = np.random.choice([-1, 1], 3)
    #     self.force_vector = np.random.uniform(0.5, 1, 3) * sign * max_force # Newton. 3D 400N is 700N max

        # if self.apply_torques and np.random.random() < rand_torque_prob and not self.torque_applied and self.force_window <= 0:
        #     max_torque = 90
        #     self.force_window = np.random.randint(60, 100)
        #     self.torque_vector = np.zeros(3)
        #     self.torque_vector[1] = np.random.choice([-1, 1], 1) * np.random.uniform() * max_torque
        #     self.force_vector = np.zeros(3)
        #     self.torque_applied = True





    def get_action_mirror_indices(self):
        return self.robot.motor_mirror_indices

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot.robot_state_mirror_indices
        mirror_inds += [len(mirror_inds)] # height commands #negative can also set to 0 in PPO training
        # mirror_inds += [len(mirror_inds)] 
        mirror_inds += [len(mirror_inds) + 3, -(len(mirror_inds) + 4), len(mirror_inds) + 5, len(mirror_inds), -(len(mirror_inds) + 1), len(mirror_inds) + 2]
        # mirror_inds += [len(mirror_inds)] * self.dummy_count
        return mirror_inds

    def _init_interactive_key_bindings(self):
        
        self.input_keys_dict["q"] = {
            "description": "increment cmd height",
            "func": lambda self: setattr(self, "cmd_height", self.cmd_height + 0.01)
        }
        self.input_keys_dict["a"] = {
            "description": "decrement cmd height",
            "func": lambda self: setattr(self, "cmd_height", self.cmd_height - 0.01)
        }
        self.input_keys_dict["e"] = {
            "description": "increase right arm z",
            "func": lambda self: setattr(self, "cmd_r_arm_z", self.cmd_r_arm_z + 0.01)
        }
        self.input_keys_dict["d"] = {
            "description": "decrement right arm z",
            "func": lambda self: setattr(self, "cmd_r_arm_z", self.cmd_r_arm_z - 0.01)
        }
        self.input_keys_dict["t"] = {
            "description": "increase right arm y",
            "func": lambda self: setattr(self, "cmd_r_arm_y", self.cmd_r_arm_y + 0.01)
        }
        self.input_keys_dict["f"] = {
            "description": "decrement right arm y",
            "func": lambda self: setattr(self, "cmd_r_arm_y", self.cmd_r_arm_y - 0.01)
        }
        self.input_keys_dict["w"] = {
            "description": "increase right arm x",
            "func": lambda self: setattr(self, "cmd_r_arm_x", self.cmd_r_arm_x + 0.01)
        }
        self.input_keys_dict["s"] = {
            "description": "decrement right arm x",
            "func": lambda self: setattr(self, "cmd_r_arm_x", self.cmd_r_arm_x - 0.01)
        }
        self.input_keys_dict["i"] = {
            "description": "increase left arm z",
            "func": lambda self: setattr(self, "cmd_l_arm_z", self.cmd_l_arm_z + 0.01)
        }
        self.input_keys_dict["k"] = {
            "description": "decrement left arm z",
            "func": lambda self: setattr(self, "cmd_l_arm_z", self.cmd_l_arm_z - 0.01)
        }
        self.input_keys_dict["u"] = {
            "description": "increase left arm y",
            "func": lambda self: setattr(self, "cmd_l_arm_y", self.cmd_l_arm_y - 0.01)
        }
        self.input_keys_dict["j"] = {
            "description": "decrement left arm y",
            "func": lambda self: setattr(self, "cmd_l_arm_y", self.cmd_l_arm_y + 0.01)
        }
        self.input_keys_dict["o"] = {
            "description": "increase left arm x",
            "func": lambda self: setattr(self, "cmd_l_arm_x", self.cmd_l_arm_x + 0.01)
        }
        self.input_keys_dict["l"] = {
            "description": "decrement left arm x",
            "func": lambda self: setattr(self, "cmd_l_arm_x", self.cmd_l_arm_x - 0.01)
        }
        def zero_command(self):
            self.cmd_height = 0.8
            self.cmd_r_arm_x = 0.25
            self.cmd_r_arm_y = -0.3
            self.cmd_r_arm_z = 0.9
            self.cmd_l_arm_x = 0.25
            self.cmd_l_arm_y = 0.3
            self.cmd_l_arm_z = 0.9
        self.input_keys_dict["0"] = {
            "description": "reset all height command to nominal",
            "func": zero_command,
        }

    def _update_control_commands_dict(self):
        self.control_commands_dict["cmd_height"] = self.cmd_height
        self.control_commands_dict["cmd_r_arm_x"] = self.cmd_r_arm_x
        self.control_commands_dict["cmd_r_arm_y"] = self.cmd_r_arm_y
        self.control_commands_dict["cmd_r_arm_z"] = self.cmd_r_arm_z
        self.control_commands_dict["cmd_l_arm_x"] = self.cmd_l_arm_x
        self.control_commands_dict["cmd_l_arm_y"] = self.cmd_l_arm_y
        self.control_commands_dict["cmd_l_arm_z"] = self.cmd_l_arm_z

    @staticmethod
    def get_env_args():
        return {
            "robot-name"         : ("cassie", "Which robot to use (\"cassie\" or \"digit\")"),
            "simulator-type"     : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\" or \"ar\")"),
            "terrain"            : ("", "What terrain to train with (default is flat terrain)"),
            "policy-rate"        : (50, "Rate at which policy runs in Hz"),
            "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
            "state-noise"        : ([0,0,0,0,0,0], "Amount of noise to add to proprioceptive state."),
            "state-est"          : (False, "Whether to use true sim state or state estimate. Only used for libcassie sim."),
            "reward-name"        : ("stand_reward", "Which reward to use"),
            "integral-action"    : (False, "Whether to use integral action in the clock (default is False)"),
        }

