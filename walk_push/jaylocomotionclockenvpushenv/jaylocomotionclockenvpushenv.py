import numpy as np

from env.util.periodicclock import PeriodicClock
from env.genericenv import GenericEnv
from util.colors import FAIL, WARNING, ENDC


class JayLocomotionClockEnvPushEnv(GenericEnv):
    """This is the base clock locomotion env. It inherits most of the functionality from GenericEnv,
    but adds clock functionality.
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
        clock_type: str,
        full_clock: bool = False,
        full_gait: bool = False,
        integral_action: bool = False,
        **kwargs,
    ):
        # Check clock types
        assert clock_type == "linear" or clock_type == "von_mises", \
            f"{FAIL}LocomotionClockEnv received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"
        if full_gait and not full_clock:
            raise NotImplementedError("Training with full gait only works with full clock.")

        super().__init__(
            robot_name=robot_name,
            reward_name=reward_name,
            simulator_type=simulator_type,
            terrain=terrain,
            policy_rate=policy_rate,
            dynamics_randomization=dynamics_randomization,
            state_noise=state_noise,
            state_est=state_est,
            integral_action=integral_action,
            **kwargs,
        )

        # Command randomization ranges
        self._x_velocity_bounds = [-1.0, 2.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._turn_rate_bounds = [-0.4, 0.4] # rad/s
        self._randomize_commands_bounds = [50, 250] # in episode length

        # Initialize commands
        self.x_velocity = 0
        self.y_velocity = 0
        self.turn_rate = 0

        # Clock variables
        self.clock_type = clock_type
        self.full_clock = full_clock

        l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[0]])
        r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[1]])
        l_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[0]])
        r_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[1]])
        self.prev_contact = np.array([l_foot_force, r_foot_force]) > 0.1
        self.min_z_foot_vel = [l_foot_vel, r_foot_vel]
        self.foot_z_pos_bonus = [0,0]
        self.feet_air_time = np.array([0,0])
        self.apply_force = 0
        self.pre_base_adr = 0
        self.dummy_count = 21
        # Command randomization rangest_in_contact
        self.full_gait = full_gait
        if self.full_gait:
            self._swing_ratio_bounds = [0.4, 0.7]
            self._period_shift_bounds = [-0.5, 0.5]
            self._cycle_time_bounds = [0.6, 1.0]
        else:
            self._swing_ratio_bounds = [0.5, 0.5]
            self._period_shift_bounds = [0.5, 0.5]
            self._cycle_time_bounds = [0.7, 0.7]

        # Initialize clock for now. Will be randomized in reset()
        self.clock = PeriodicClock(0.8, 1 / self.default_policy_rate, [0.5, 0.5], [0.0, 0.5])
        self.clock._phase = 0
        self.clock._von_mises_buf = None

        # Only check obs if this envs is inited, not when it is parent:
        if self.__class__.__name__ == "LocomotionClockEnv" and self.simulator_type != "ar_async":
            self.check_observation_action_size()

    @property
    def observation_size(self):
        observation_size = super().observation_size
        observation_size += 3 # XY velocity and turn command
        observation_size += 2 # swing ratio
        observation_size += 2 # period shift
        observation_size += 2 # input clock sin/cos
        if self.full_clock:
            observation_size += 2 # 2 more
        # observation_size += self.dummy_count
        return observation_size

    @property
    def extra_input_names(self):
        extra_input_names = ['x-velocity', 'y-velocity', 'turn-rate']
        extra_input_names += ['swing-ratio-left', 'swing-ratio-right', 'period-shift-left', 'period-shift-right']
        if self.full_clock:
            extra_input_names += ['clock-sin-left', 'clock-cos-left', 'clock-sin-right', 'clock-cos-right']
        else:
            extra_input_names += ['clock-sin', 'clock-cos']
        # extra_input_names += [0] * self.dummy_count
        return extra_input_names

    def reset(self, interactive_evaluation=False):
        self.randomize_clock(init=True)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        self.randomize_commands_at = np.random.randint(*self._randomize_commands_bounds)
        self.randomize_commands()

        self.reset_simulation()
        self.randomize_base_orientation()

        self.interactive_evaluation = interactive_evaluation
        if self.interactive_evaluation:
            self._update_control_commands_dict()

        # Reset env counter variables
        self.traj_idx = 0
        self.max_foot_vel = 0
        self.last_action = None

        return self.get_state()

    def sample_from_body(self):
        # body_id: 1~5 and 20~23
        body_id = np.random.choice(np.concatenate((np.arange(1, 6), np.arange(20, 24), [16], [34])))
        x = np.random.uniform(-200, 200)
        y = np.random.uniform(-150, 150)
        z = np.random.uniform(-100, 100)
        return body_id, [x, y, z]

    def sample_from_arm(self):
        # body_id: 17~19 and 35~37
        body_id = np.random.choice(np.concatenate((np.arange(17, 20), np.arange(35, 38))))
        x = np.random.uniform(-150, 150)
        y = np.random.uniform(-100, 100)
        z = np.random.uniform(-100, 130)
        return body_id, [x, y, z]

    def sample_from_leg(self):
        # body_id: 6~15 and 24~33
        body_id = np.random.choice(np.concatenate((np.arange(6, 16), np.arange(24, 34))))
        x = np.random.uniform(-80, 80)
        y = np.random.uniform(-80, 80)
        z = np.random.uniform(-80, 80)
        return body_id, [x, y, z]

    def randomize_commands(self):
        self.x_velocity = np.random.uniform(*self._x_velocity_bounds)
        self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
        choices = ['in-place-stand', 'in-place-turn', 'walk', 'walk-sideways', 'walk-turn']
        mode = np.random.choice(choices, p=[0.2, 0.2, 0.3, 0.1, 0.2])
        match mode:
            case 'in-place-stand':
                self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
            case 'in-place-turn':
                self.x_velocity, self.y_velocity = 0, 0
            case 'walk':
                self.turn_rate = 0
            case 'walk-sideways':
                self.x_velocity, self.turn_rate = 0, 0
        # Clip to avoid useless commands
        if np.abs(self.x_velocity) <= 0.1:
            self.x_velocity = 0
        if np.abs(self.y_velocity) <= 0.1:
            self.y_velocity = 0
        if np.abs(self.turn_rate) <= 0.1:
            self.turn_rate = 0
        
    def randomize_clock(self, init=False):
        phase_add = 1 / self.default_policy_rate
        if init:
            swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
            swing_ratios = [swing_ratio, swing_ratio]
            if np.random.random() < 0.3 and self.full_gait: # 30% chance of rand shifts
                period_shifts = [0   + np.random.uniform(*self._period_shift_bounds),
                                 0.5 + np.random.uniform(*self._period_shift_bounds)]
            else:
                period_shifts = [0, 0.5]
            self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
            self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        else:
            swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
            self.clock.set_swing_ratios([swing_ratio, swing_ratio])
            if np.random.random() < 0.3 and self.full_gait: # 30% chance of rand shifts
                period_shifts = [0   + np.random.uniform(*self._period_shift_bounds),
                                 0.5 + np.random.uniform(*self._period_shift_bounds)]
            else:
                period_shifts = [0, 0.5]
            self.clock.set_period_shifts(period_shifts)
            self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
            self.clock.set_cycle_time(self.cycle_time)

    def step(self, action: np.ndarray):
        # print("right-hand = ",self.sim.get_site_pose('right-hand')[0])
        # print("right-hand = ",self.sim.get_site_pose('right-hand')[1])
        # print("right-hand = ",self.sim.get_site_pose('right-hand')[2])
        self.policy_rate = self.default_policy_rate
        if self.dynamics_randomization:
            self.policy_rate += np.random.randint(0, 6)

        # Offset global zero heading by turn rate per policy step
        self.orient_add += self.turn_rate / self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps, integral_action=self.integral_action)

        # Reward for taking current action before changing quantities for new state
        self.compute_reward(action)

        # Increment episode counter and update previous attributes
        self.traj_idx += 1
        self.last_action = action

        # Increment clock at last for updating s'
        self.clock.increment()

        #print(self.clock.get_swing_ratios())
        
        #reset force place
        if self.traj_idx % 250 == 0 or self.traj_idx == 1:
            
            if self.traj_idx < 2000:
                self.sim.data.xfrc_applied[self.pre_base_adr, 0:3] = np.zeros(3)

            self.push_it()

        # Randomize commands
        if self.traj_idx % self.randomize_commands_at == 0 and self.interactive_evaluation:
            self.randomize_commands()
            if self.full_gait:
                self.randomize_clock()


        if   self.interactive_evaluation:
            if self.push_start_time <= self.traj_idx < self.push_start_time + self.push_duration:
                self.sim.data.xfrc_applied[self.base_adr, 0:3] = self.push_force
                self.pre_base_adr = self.base_adr
                
                self.apply_force = 1
            elif self.traj_idx == self.push_start_time + self.push_duration and self.traj_idx < 2000:
                
                self.sim.data.xfrc_applied[self.pre_base_adr, 0:3] = np.zeros(3)
                self.apply_force = 0

        return self.get_state(), self.reward, self.compute_done(), {'rewards': self.reward_dict}

    def push_it(self):

        # [1~5,-200~200,150,-100~100][16, -200~200,150,-100~100][17~19,-150~150,-100~100,-100~130][6~15, 100~100,100~100,100~100]
        group_choice = np.random.choice([1, 2, 3], p = [0.55, 0.15, 0.30])
        if group_choice == 1:
            self.base_adr, self.push_force = self.sample_from_body()
        elif group_choice == 2:
            self.base_adr, self.push_force = self.sample_from_arm()
        elif group_choice == 3:
            self.base_adr, self.push_force = self.sample_from_leg()

        self.push_duration = np.random.randint(20, 50)
        self.push_start_time = self.traj_idx
        self.push_start_time += np.random.randint(50, 200)
        
        


    def hw_step(self):
        self.orient_add += self.turn_rate / self.default_policy_rate
        self.clock.increment()

    def _get_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()
        return np.concatenate((
            self.get_robot_state(),
            [self.x_velocity, self.y_velocity, self.turn_rate],
            [self.clock.get_swing_ratios()[0], 1 - self.clock.get_swing_ratios()[0]],
            self.clock.get_period_shifts(),
            input_clock,
            # np.zeros(self.dummy_count),
        ))

    def get_action_mirror_indices(self):
        return self.robot.motor_mirror_indices

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot.robot_state_mirror_indices
        mirror_inds += [len(mirror_inds), -(len(mirror_inds) + 1), -(len(mirror_inds) + 2)] # XY velocity commands
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)] # swing ratio
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)] # period shift
        if self.full_clock: # input clock sin/cos
            mirror_inds += [len(mirror_inds) + 2, len(mirror_inds) + 3, len(mirror_inds), len(mirror_inds) + 1]
        else:
            mirror_inds += [-len(mirror_inds), -(len(mirror_inds) + 1)]
        # mirror_inds += [len(mirror_inds)] * self.dummy_count
        return mirror_inds

    def _init_interactive_key_bindings(self):
        self.input_keys_dict["w"] = {
            "description": "increment x velocity",
            "func": lambda self: setattr(self, "x_velocity", self.x_velocity + 0.1)
        }
        self.input_keys_dict["s"] = {
            "description": "decrement x velocity",
            "func": lambda self: setattr(self, "x_velocity", self.x_velocity - 0.1)
        }
        self.input_keys_dict["d"] = {
            "description": "increment y velocity",
            "func": lambda self: setattr(self, "y_velocity", self.y_velocity + 0.1)
        }
        self.input_keys_dict["a"] = {
            "description": "decrement y velocity",
            "func": lambda self: setattr(self, "y_velocity", self.y_velocity - 0.1)
        }
        self.input_keys_dict["e"] = {
            "description": "decrease turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate - 0.1)
        }
        self.input_keys_dict["q"] = {
            "description": "increase turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate + 0.1)
        }
        self.input_keys_dict["o"] = {
            "description": "increase clock cycle time",
            "func": lambda self: setattr(self.clock, "_cycle_time", min(self.clock._cycle_time + 0.01, self._cycle_time_bounds[1]))
        }
        self.input_keys_dict["u"] = {
            "description": "decrease clock cycle time",
            "func": lambda self: setattr(self.clock, "_cycle_time", max(self.clock._cycle_time - 0.01, self._cycle_time_bounds[0]))
        }
        self.input_keys_dict["-"] = {
            "description": "increase swing ratio",
            "func": lambda self: setattr(self.clock, "_swing_ratios", np.ones(2) * min(self.clock._swing_ratios[0] + 0.1, self._swing_ratio_bounds[1]))
        }
        self.input_keys_dict["="] = {
            "description": "decrease swing ratio",
            "func": lambda self: setattr(self.clock, "_swing_ratios", np.ones(2) * max(self.clock._swing_ratios[0] - 0.1, self._swing_ratio_bounds[0]))
        }
        self.input_keys_dict["k"] = {
            "description": "increase period shift",
            "func": lambda self: setattr(self.clock, "_period_shifts", np.array([0, min(self.clock._period_shifts[1] + 0.05, self._period_shift_bounds[1])]))
        }
        self.input_keys_dict["l"] = {
            "description": "decrease period shift",
            "func": lambda self: setattr(self.clock, "_period_shifts", np.array([0, max(self.clock._period_shifts[1] - 0.05, self._period_shift_bounds[0])]))
        }
        def zero_command(self):
            self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
            self.clock.set_cycle_time(0.8)
            self.clock.set_swing_ratios([0.5, 0.5])
            self.clock.set_period_shifts([0, 0.5])
        self.input_keys_dict["0"] = {
            "description": "reset all commands to zero",
            "func": zero_command
        }

    def _update_control_commands_dict(self):
        self.control_commands_dict["x velocity"] = self.x_velocity
        self.control_commands_dict["y velocity"] = self.y_velocity
        self.control_commands_dict["turn rate"] = self.turn_rate
        self.control_commands_dict["clock cycle time"] = self.clock._cycle_time
        self.control_commands_dict["swing ratios"] = tuple(round(x, 2) for x in (
            self.clock._swing_ratios[0], self.clock._swing_ratios[1]))
        self.control_commands_dict["period shifts"] = tuple(round(x, 2) for x in (
            self.clock._period_shifts[0], self.clock._period_shifts[1]))

    @staticmethod
    def get_env_args():
        return {
            "robot-name"      : ("cassie", "Which robot to use (\"cassie\" or \"digit\")"),
            "simulator-type"  : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\" or \"ar_async\")"),
            "terrain"         : ("", "What terrain to train with (default is flat terrain)"),
            "policy-rate"     : (50, "Rate at which policy runs in Hz"),
            "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
            "state-noise"     : ([0,0,0,0,0,0], "Amount of noise to add to proprioceptive state."),
            "state-est"       : (False, "Whether to use true sim state or state estimate. Only used for libcassie sim."),
            "reward-name"     : ("locomotion_vonmises_clock_reward", "Which reward to use"),
            "clock-type"      : ("von_mises", "Which clock to use (\"linear\" or \"von_mises\")"),
            "full-clock"      : (False, "Whether to input the full clock (sine/cosine for each leg) or just \
                                         single sine/cosine pair (default is False)"),
            "full-gait"       : (False, "Whether to train on all gait parameters or just train walking \
                                         (default is False)"),
            "integral-action" : (False, "Whether to use integral action in the clock (default is False)"),
        }
