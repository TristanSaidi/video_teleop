""" Module implements IHMEnv - the base class for in-hand manipulation"""
import multiprocessing as mp
import pickle
from collections import OrderedDict
from typing import Dict, Optional

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from sim.hand import Hand
from logger import getlogger

logger = getlogger(__name__)


class IHMEnv(gym.Env):
    """Environment for in-hand manipulation"""

    def __init__(
        self,
        sim: Hand,
        step_duration: float = 0.05,
        max_episode_length: int = 1000,
        max_dq: float or np.ndarray = 0.025,
        action_scaling_type: str = 'clip',
        discrete_action: bool = False,
        discretization_steps: int = 5,
        randomize_initial_state: bool = False,
        num_fingers: int = 5,
    ):

        """
        Args:
            sim: Hand simulation object to be wrapped
            step_duration: Simulation duration for each step
            max_episode_length: Episode forced to terminate
            after max_episode_length steps
            max_dq: Upper limit on the maximum setpoint change in a step
                if normalization_method is clip, maximum standard deviation 
                if normalization_method is std, maximum after scale
                if normalization_method is 'scale'
            normalization_method: 'clip', 'std' or 'scale'
            discrete_action: Switch between a discrete and a
                continuous action space
            discretization_steps: Number of discrete actions to
                discretize the actions each action dimension
            randomize_initial_state: If True, environment uses
                the state sampler to randomize initial state
        """
        logger.info("Initializing env ...")

        # Simulation
        self.sim = sim
        self.step_duration = step_duration
        self.max_episode_length = max_episode_length
        self._step = 0

        # Action discretization
        self._discrete_action = discrete_action
        self.discretization_steps = discretization_steps
        if discrete_action:
            assert (
                discretization_steps % 2 != 0
            ), "discretization steps \
                must be odd"

        # Control clipping
        if isinstance(max_dq, float):
            self.max_dq = max_dq * np.ones(self.action_dim)
        elif isinstance(max_dq, np.ndarray):
            # assert (
            #     max_dq.shape[0] == self.action_dim
            # ), "length of max_dq \
            #      must equal action_dim"
            self.max_dq = max_dq.reshape(-1)
        else:
            raise ValueError("Invalid type specified for max_dq")
        self.action_scaling_type = action_scaling_type

        self.num_fingers = num_fingers

        self._randomize_initial_state = randomize_initial_state
        logger.info("Creating a copy of simulation for initial state sampler..")

    @property
    def action_space(self) -> gym.Space:
        """Action space

        Box or MultiDiscrete action space

        """
        if self._discrete_action:
            return spaces.MultiDiscrete([self.discretization_steps] * self.action_dim)
        else:
            return spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        """Observation space

        Return the observation space consisting of hand, object and
        contact information

        """
        obs = self._get_obs()
        space = {}
        for key, val in obs.items():
            space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=val.shape)

        return spaces.Dict(space)

    def reset(self, **kwargs):
        self._step = 0
        self.sim.reset()
        return self._get_obs()

    def render(self, mode="human"):
        return self.sim.render(mode)

    def close(self):
        # Close resources and delete references to avoid memory leak
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def seed(self, seed=None):
        # Save the seed so we can re-seed during un-pickling
        self._seed = seed

        # Hash the seed to avoid any correlations
        seed = seeding.hash_seed(seed)

        # Seed environment components with randomness
        seeds = [seed]
        seeds.extend(self.sim.seed(seed))

        return seeds

    def step(self, action_dict):
        """ expects cartesian coordinates of fingertips and desired pose of UR5 ee"""
        hand_action = action_dict["hand"]
        arm_action = action_dict["arm"]
        # parse hand action - expects cartesian coordinates of fingertips
        assert hand_action.shape == (self.num_fingers, 3), f"Cartesian action for hand must be of shape ({self.num_fingers}, 3)"
        assert arm_action.shape == (4, 4), "Action for arm must be of shape (4, 4)"
        dq_hand = self._compute_dq_cartesian_hand(hand_action)
        dq_arm = self._compute_dq_cartesian_arm(arm_action)

        dq = np.concatenate((dq_hand, dq_arm))
        # parse arm action - expects joint positions
        self.sim.update_joint_setpoint(dq)
        self.sim.advance(self.step_duration)
        obs = self._get_obs()
        self._step += 1
        ncontacts = len(self.sim.sense_ftip_contacts(add_noise=False))
        done = self._step >= self.max_episode_length
        rew = self._reward()
        return obs, rew, done, {"done": done, "reward": rew}

    def _compute_dq(self, action):
        """Compute joint setpoint delta (dq) for action"""
        if not self._discrete_action:
            if self.action_scaling_type == 'clip':
                action = np.clip(action, a_min=-1, a_max=1)
            elif self.action_scaling_type == 'std':
                if np.std(action) > 1:
                    action = action / np.std(action)
            elif self.action_scaling_type == 'scale':
                if np.abs(action).max() > 1:
                    action = action / np.abs(action).max()
            dq = np.multiply(self.max_dq, action)
        else:
            dq = np.array(
                [
                    (2.0 * a_i / float(self.discretization_steps - 1) - 1.0) * max_dq
                    for a_i, max_dq in zip(action, self.max_dq)
                ]
            )
        return dq

    def _compute_dq_cartesian_arm(self, pose):
        """Compute joint setpoint delta (dq) for action"""
        # obtain desired joint positions via ik
        current_joint_pos = self.sim.arm_joint_pos
        
        desired_pose = self.sim.check_ik_pos_arm(pose)
        desired_joint_pos = self.sim.compute_ik_arm(desired_pose, current_joint_pos)

        # desired_joint_pos = self._clamp_arm_action(desired_joint_pos)

        action = desired_joint_pos - current_joint_pos
        if not self._discrete_action:
            if self.action_scaling_type == 'clip':
                action = np.clip(action, a_min=-1, a_max=1)
            elif self.action_scaling_type == 'std':
                if np.std(action) > 1:
                    action = action / np.std(action)
            elif self.action_scaling_type == 'scale':
                if np.abs(action).max() > 1:
                    action = action / np.abs(action).max()
            dq = np.multiply(self.max_dq[-len(action):], action)
        return dq * 0.5

    def _compute_dq_cartesian_hand(self, pos):
        """Compute joint setpoint delta (dq) for action (treated as cartesian coords of end-effector)"""
        current_joint_pos = self.sim.hand_joint_pos
        pos = self.sim.check_ik_pos_hand(pos)
        desired_joint_pos = self.sim.compute_ik_hand(pos, roll=False)
        # clamp dq to be within join limits
        hand_joint_ctrl_range_low, hand_joint_ctrl_range_high = self.sim._get_hand_joint_ctrl_range()
        desired_joint_pos = np.clip(desired_joint_pos, hand_joint_ctrl_range_low, hand_joint_ctrl_range_high)
        action = desired_joint_pos - current_joint_pos
        if not self._discrete_action:
            if self.action_scaling_type == 'clip':
                action = np.clip(action, a_min=-1, a_max=1)
            elif self.action_scaling_type == 'std':
                if np.std(action) > 1:
                    action = action / np.std(action)
            elif self.action_scaling_type == 'scale':
                if np.abs(action).max() > 1:
                    action = action / np.abs(action).max()
            if isinstance(self.max_dq, float):
                dq = np.multiply(self.max_dq, action)
            else:
                dq = np.multiply(self.max_dq[0:len(action)], action)
        else:
            dq = np.array(
                [
                    (2.0 * a_i / float(self.discretization_steps - 1) - 1.0) * max_dq
                    for a_i, max_dq in zip(action, self.max_dq)
                ]
            )
        return dq

    def _get_obs(self) -> OrderedDict:
        """Return observation

        For attributes which can be noisy ex. hand_joint_position,
        contact_position, contact_normal etc,. both the "accurate" and
        noisy versions are inluded in the observation dictionary. The
        noisy version is the one with suffix "_noise". Helpful towards
        using assymmetric actor-critic architectures.

        """

        obs = OrderedDict()

        # Hand
        obs["hand_joint_position"] = self.sim.hand_joint_pos.copy()
        obs["hand_joint_position_noise"] = self.sim.hand_joint_pos_noise.copy()
        obs["hand_joint_velocity"] = self.sim.hand_joint_vel.copy()
        obs["hand_joint_control"] = self.sim.hand_joint_ctrl.copy()
        obs["hand_joint_setpoint"] = self.sim.hand_joint_setpoint.copy()

        # Arm
        obs["arm_position"] = self.sim.arm_joint_pos.copy()

        # Object
        obs["object_pose"] = self.sim.object_pose.copy()
        obs["object_position"] = self.sim.object_pos.copy()
        obs["object_orientation"] = self.sim.object_rot.copy()
        obs["object_angular_velocity"] = self.sim.object_ang_vel.copy()

        # Contact
        nftips = len(self.sim.ftip_links)
        prefix = "contact_"
        attrdims = {
            "position": 3,
            "force": 3,
            "normal": 3,
            "force_magnitude": 1,
            "wrench": 6,
        }
        for attr, dim in attrdims.items():
            for suffix in ["", "_noise"]:
                obskey = prefix + attr + suffix
                obs[obskey] = np.zeros(dim * nftips)
        for add_noise in [False, True]:
            prefix = "contact_"
            suffix = "_noise" if add_noise else ""
            contacts = self.sim.sense_ftip_contacts(add_noise)
            for i, ftip in enumerate(self.sim.ftip_links):
                for contact in contacts:
                    if contact["link"] == ftip:
                        for attr, dim in attrdims.items():
                            obskey = prefix + attr + suffix
                            obs[obskey][dim * i : dim * (i + 1)] = contact[attr].copy()
                        break
        return obs

    def _reward(self):
        return 0.0

    @property
    def action_dim(self) -> int:
        """Action dimension"""
        return self.sim.num_hand_joints