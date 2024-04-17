""" Implements the base environment for in-hand manipulation
"""

import copy
from typing import Dict, List, Optional, Union

import numpy as np
import pickle
from gym.utils import seeding
from logger import getlogger
from .sim import Sim, State
from .tactile_sensor import TactileSensor
from .trajplanner import TrajPlanner

from ikpy import chain

logger = getlogger(__name__)

__all__ = ["Hand"]


class Hand(Sim):
    """

    Base class for hand simulation

    (1) Implements an inferface to MuJuCo simulation of the hand.
    (2) Simulate trajectory planning used to generate intermediate
    setpoints in servo control on-top of the controller in MuJoCo.
    (3) Simulate tactile sensing on finger tips

    NOTE: Support for torque controlled actuators is yet to be added.

    """

    def __init__(
        self,
        *,
        model: Union[str, bytes] = None,
        ur5_kinematics: chain.Chain,
        state: State = None,
        default_hand_joint_pos: Optional[np.ndarray] = None,
        default_arm_joint_pos: Optional[np.ndarray] = None,
        default_object_pose: Optional[np.ndarray] = None,
        table_height: float = 0.1,
        simulation_settling_time: float = 1,
        trajplanner_params: dict = {
            "frequency": 250,
            "profile": "rectangle",
            "vel": 1,
            "acc": 100,
        },
        tactile_sensor_params: dict = {"frequency": 40},
        hand_joint_position_noise: float = 0.1,
    ) -> None:
        """

        Args:
            default_hand_joint_pos: Hand joints are initialized in this pose.
            default_object_pose: Object is initialized in this pose by default
            simulation_settling_time: Controls the settling time during reset
            trajplanner_params: Arguments dict to trajplanner
            tactile_sensor_params: Arguments dict to tactile_sensor
            hand_joint_position_noise: Stddev of gaussian noise in sensing
                hand joint position

        """

        super().__init__(model=model, state=state)
        self._hand_joints = self._get_hand_joints()
        self._arm_joints = self._get_arm_joints()
        self._num_hand_joints = len(self._hand_joints)
        self._num_arm_joints = len(self._arm_joints)

        self._link_lengths = {
            f"finger{i+1}": self._get_link_lengths(f"finger{i+1}") for i in range(self.num_ftips)
        }

        # Set hand and object pose
        self._default_hand_joint_pos = default_hand_joint_pos
        self._default_arm_joint_pos = default_arm_joint_pos
        self._default_object_pose = default_object_pose
        self._simulation_settling_time = simulation_settling_time
        self._default_state = None
        self._default_state = pickle.loads(pickle.dumps(self.reset()))

        self.table_height = table_height

        # Trajplanner
        self._trajplanner_params = trajplanner_params
        self._setup_trajplanners(trajplanner_params)

        # Contacts
        self._ftip_contacts = []
        self._tactile_sensor_params = tactile_sensor_params
        self._setup_tactile_sensor(tactile_sensor_params)

        # Random number generators
        self._np_random = {}
        for name in ["hand_joint_pos"]:
            self._np_random[name] = np.random.default_rng()
        self._hand_joint_position_noise = hand_joint_position_noise

        self.ur5_kinematics = ur5_kinematics
        
    def reset(self) -> Dict:
        """Reset to state with default hand joint pos and object pose"""
        logger.info("Resetting simulation to default state")
        # Disable callbacks
        callbacks = [_ for _ in self._callbacks]
        self._callbacks = []

        # Reset
        if self._default_state is None:
            if self._default_hand_joint_pos is not None:
                self._set_hand_joint_pos(self._default_hand_joint_pos)
                self._set_hand_joint_ctrl(self._default_hand_joint_pos)
            if self._default_arm_joint_pos is not None:
                self._set_arm_joint_pos(self._default_arm_joint_pos)
                self._set_arm_joint_ctrl(self._default_arm_joint_pos)
            if self._sim_has_object() and self._default_hand_joint_pos is not None:
                self._set_object_pose(self._default_object_pose)
            self.advance(self._simulation_settling_time)
        else:
            self.set_state(self._default_state)
            self.set_hand_joint_setpoint(self._default_state.ctrl)

        # Enable callbacks
        self._callbacks = callbacks

        return self.get_state()

    def set_gravity(self, gravity: np.ndarray):
        """Set gravity
        Args: gravity, (3,) array
        """
        self._model.opt.gravity[:] = gravity
        self._sim.forward()

    def set_actuator_gains(self, kp):
        """Set actuator gain
        Args: kp
        """
        print(
            "Warning: changing actuator gain on-the-fly post compilation does not seem to take effect"
        )
        self._model.actuator_gainprm[:, 0] = kp
        self._sim.forward()

    def set_hand_joint_setpoint(self, setpoint):
        """Set hand joint setpoint
        trajplanner plans a new a trajectory to reach this setpoint
        for every call to this method
        """
        return self._set_setpoint(setpoint)

    def set_hand_joint_ctrl(self, pos):
        return self._set_hand_joint_ctrl(pos)

    @property
    def num_hand_joints(self):
        """num_joints in hand"""
        return self._num_hand_joints

    def update_joint_setpoint(self, dq):
        return self._update_setpoint(dq)

    @property
    def hand_joint_setpoint(self):
        """Return joint setpoint of the trajplanner"""
        return self._get_setpoint()

    @property
    def hand_joint_pos(self):
        """Return current joint position"""
        return self._get_hand_joint_pos()

    @property
    def arm_joint_pos(self):
        """Return current joint position"""
        return self._get_arm_joint_pos()

    @property
    def hand_joint_pos_noise(self):
        """Return current joint position"""
        joint_pos = self._get_hand_joint_pos()
        joint_pos += self._np_random["hand_joint_pos"].normal(
            scale=self._hand_joint_position_noise * np.ones_like(joint_pos),
        )
        return joint_pos

    @property
    def hand_joint_ctrl(self):
        """Return joint target use by MuJoCo trajplanner"""
        return self._get_hand_joint_ctrl()

    @property
    def hand_joint_vel(self):
        """Return hand joint velocity"""
        return self._get_hand_joint_vel()

    @property
    def object_pos(self):
        """Return object position"""
        return self._get_object_pos()

    @property
    def object_rot(self):
        """Return object orientation as a quat"""
        return self._get_object_quat()

    @property
    def object_pose(self):
        """Return object pose"""
        return np.hstack([self._get_object_pos(), self._get_object_quat()])

    @property
    def object_ang_vel(self):
        """Returns object angular velocity"""
        return self._get_object_angvel()

    @property
    def num_ftips(self):
        return len(self._get_ftip_links())

    @property
    def ftip_links(self):
        return self._get_ftip_links()

    def sense_ftip_contacts(self, add_noise=False):

        contacts = self._sense_ftip_contacts(add_noise)

        return contacts

    def get_ftip_contacts(self):
        return self._get_ftip_contacts()

    def get_hand_proximal_links(self):
        return self._get_hand_proximal_links()

    def get_hand_joints(self):
        return self._get_hand_joints()

    def get_arm_joints(self):
        return self._get_arm_joints()

    def get_link_lengths(self, finger):
        return self._get_link_lengths(finger)

    def get_ncontacts(self):
        return self._get_ncontacts()

    def set_object_pose(self, pose):
        return self._set_object_pose(pose)

    def set_hand_joint_pos(self, pos):
        return self._set_hand_joint_pos(pos)

    # --------

    # Hand
    def _get_hand_joints(self) -> List[str]:
        hand_actuators = [actuator for actuator in self._model.actuator_names if "finger" in actuator]
        return [actuator.replace("_actuator", "") for actuator in hand_actuators]

    def _get_arm_joints(self) -> List[str]:
        arm_actuators = [actuator for actuator in self._model.actuator_names if "finger" not in actuator]
        return [actuator.replace("_actuator", "") for actuator in arm_actuators]

    def _get_hand_proximal_joints(self) -> List[str]:
        return [joint for joint in self._hand_joints if "prox" in joint]

    def _get_hand_proximal_links(self) -> List[str]:
        return [link for link in self._model.body_names if "proximal" in link]

    def _get_finger_links(self, finger: str) -> List[str]:
        return [link for link in self._model.body_names if finger in link]

    def _get_ftip_links(self) -> List[str]:
        return [link for link in self._model.body_names if "distal" in link]

    def _get_hand_joint_pos(self) -> np.ndarray:
        pos = np.zeros((self._num_hand_joints,))
        for i, joint in enumerate(self._hand_joints):
            pos[i] = self._sim.data.get_joint_qpos(joint)
        return pos

    def _get_arm_joint_pos(self) -> np.ndarray:
        pos = np.zeros((self._num_arm_joints,))
        for i, joint in enumerate(self._arm_joints):
            pos[i] = self._sim.data.get_joint_qpos(joint)
        return pos

    def _get_hand_joint_pos_ref(self) -> np.ndarray:
        """
        Return the joint value corresponding to the initial model
        configuration. This is set by "ref" attribute of the joint.
        We use it to simulate joint ref error common on real hardware.

        Documentation from MuJoCo XML Reference:
        The reference position or angle of the joint. This attribute is
        only used for slide and hinge joints. It defines the joint value
        corresponding to the initial model configuration. The amount of
        spatial transformation that the joint applies at runtime equals
        the current joint value stored in mjData.qpos minus this
        reference value stored in mjModel.qpos0. The meaning of these
        vectors was discussed in the Stand-alone section in the
        Overview chapter.

        """
        pos = np.zeros((self._num_hand_joints,))
        for i, joint in enumerate(self._hand_joints):
            jointid = self._model.joint_name2id(joint)
            pos[i] = self.model.qpos0[jointid]
        return pos

    def _set_hand_joint_pos(self, pos: np.ndarray) -> None:
        for i, joint in enumerate(self._hand_joints):
            self._sim.data.set_joint_qpos(joint, pos[i])
        self._sim_data_dirty = True

    def _set_arm_joint_pos(self, pos: np.ndarray) -> None:
        for i, joint in enumerate(self._arm_joints):
            self._sim.data.set_joint_qpos(joint, pos[i])
        self._sim_data_dirty = True

    def _get_hand_joint_vel(self) -> np.ndarray:
        vel = np.zeros((self._num_hand_joints,))
        for i, joint in enumerate(self._hand_joints):
            vel[i] = self._sim.data.get_joint_qvel(joint)
        return vel

    def _set_hand_joint_vel(self, vel: np.ndarray) -> None:
        for i, joint in enumerate(self._hand_joints):
            self._sim.data.set_joint_qvel(joint, vel[i])
        self._sim_data_dirty = True

    def _get_hand_joint_ctrl(self) -> np.ndarray:
        ctrl = np.zeros((self._num_hand_joints,))
        hand_actuators = [actuator for actuator in self._model.actuator_names if "finger" in actuator]
        for i, actuator in enumerate(hand_actuators):
            index = self._model.actuator_name2id(actuator)
            ctrl[i] = self._sim.data.ctrl[index]
        return ctrl

    def _get_arm_joint_ctrl(self) -> np.ndarray:
        ctrl = np.zeros((self._num_arm_joints,))
        arm_actuators = [actuator for actuator in self._model.actuator_names if "finger" not in actuator]
        for i, actuator in enumerate(arm_actuators):
            index = self._model.actuator_name2id(actuator)
            ctrl[i] = self._sim.data.ctrl[index]
        return ctrl

    def _set_hand_joint_ctrl(self, ctrl: np.ndarray) -> None:
        hand_actuators = [actuator for actuator in self._model.actuator_names if "finger" in actuator]
        for i, actuator in enumerate(hand_actuators):
            index = self._model.actuator_name2id(actuator)
            self._sim.data.ctrl[index] = ctrl[i]
        self._sim_data_dirty = True

    def _set_arm_joint_ctrl(self, ctrl: np.ndarray) -> None:
        arm_actuators = [actuator for actuator in self._model.actuator_names if "finger" not in actuator]
        for i, actuator in enumerate(arm_actuators):
            index = self._model.actuator_name2id(actuator)
            self._sim.data.ctrl[index] = ctrl[i]
        self._sim_data_dirty = True

    def _get_hand_joint_torque(self) -> np.ndarray:
        torque = np.zeros((self._num_hand_joints,))
        for i, actuator in enumerate(self._model.actuator_names):
            index = self._model.actuator_name2id(actuator)
            torque[i] = self._sim.data.actuator_force[index]
        return torque

    def _get_hand_joint_pos_range(self) -> np.ndarray:
        low = np.zeros((self._num_hand_joints,))
        high = np.zeros((self._num_hand_joints,))
        hand_actuators = [actuator for actuator in self._model.actuator_names if "finger" in actuator]
        for i, actuator in enumerate(hand_actuators):
            index = self._model.joint_name2id(joint)
            low[i] = self._model.jnt_range[index, 0]
            high[i] = self._model.jnt_range[index, 1]
        return low, high

    def _get_hand_joint_ctrl_range(self) -> np.ndarray:
        low = np.zeros((self._num_hand_joints,))
        high = np.zeros((self._num_hand_joints,))
        hand_actuators = [actuator for actuator in self._model.actuator_names if "finger" in actuator]
        for i, actuator in enumerate(hand_actuators):
            index = self._model.actuator_name2id(actuator)
            low[i] = self._model.actuator_ctrlrange[index, 0]
            high[i] = self._model.actuator_ctrlrange[index, 1]
        return low, high

    def _get_arm_joint_ctrl_range(self) -> np.ndarray:
        low = np.zeros((self._num_arm_joints,))
        high = np.zeros((self._num_arm_joints,))
        arm_actuators = [actuator for actuator in self._model.actuator_names if "finger" not in actuator]
        for i, actuator in enumerate(arm_actuators):
            index = self._model.actuator_name2id(actuator)
            low[i] = self._model.actuator_ctrlrange[index, 0]
            high[i] = self._model.actuator_ctrlrange[index, 1]
        return low, high

    def _get_link_lengths(self, finger) -> np.ndarray:
        links = self._get_finger_links(finger)
        lengths = np.zeros(3)

        # Get first two lengths from link positions relative to parent link
        for i, link in enumerate(links[1:]):
            index = self._model.body_name2id(link)
            pos = self._model.body_pos[index]
            # Link must be aligned with y-axis
            assert pos[0] == 0 and pos[2] == 0
            lengths[i] = pos[1]
        # Get last link length from site position relative to parent link.
        # Site must be named accordingly.
        index = self._model.site_name2id(finger + "_tip")
        pos = self._model.site_pos[index]
        assert pos[0] == 0 and pos[2] == 0
        lengths[2] = pos[1]
        return lengths

    def _get_ftip_poses(self):
        """
        Return the pose of all finger tips as a dictionary.
        {
            "finger1": (pos, mat)
                    .
                    .
            "fingerN": (pos, mat)
        }

        The keys are finger names (ex. "finger1") and the values are
        tuples with position and orientation matrix

        """
        ftip_poses = {}
        for ftip in self._get_ftip_links():
            tip_pos = self.data.get_site_xpos(ftip.replace("distal", "tip"))
            tip_frame = self.data.get_site_xmat(ftip.replace("distal", "tip"))
            ftip_poses[ftip.replace("_distal", "")] = (
                tip_pos.copy(),
                tip_frame.copy(),
            )
        return ftip_poses

    # Object
    def _sim_has_object(self) -> bool:
        """Returns True if the object exists in simulation"""
        return "object" in self._model.body_names

    def _get_object_dofs(self) -> np.ndarray:
        if self._sim_has_object() is False:
            raise RuntimeError("object not loaded")
        body_index = self._model.body_name2id("object")
        num_joints = self._model.body_jntnum[body_index]
        start_index = self._model.body_jntadr[body_index]
        dofs = np.zeros(6)
        for i in range(num_joints):
            joint_name = self._model.joint_id2name(start_index + i)
            xaxis = copy.copy(self._sim.data.get_joint_xaxis(joint_name))
            joint_type = self._model.jnt_type[start_index + i]
            # Free joint
            if joint_type == 0:
                dofs = np.array([1] * 6)
            # Ball joint
            elif joint_type == 1:
                dofs[3:] += np.array([1] * 3)
            # Slide joint
            elif joint_type == 2:
                dofs[:3] += np.array([np.isclose(i, 1.0) or np.isclose(i, -1.0) for i in xaxis])
            # Hinge joint
            elif joint_type == 3:
                dofs[3:] += np.array([np.isclose(i, 1.0) or np.isclose(i, -1.0) for i in xaxis])
        return dofs

    def _set_object_pose(self, pose) -> None:
        if self._sim_has_object() is False:
            raise RuntimeError("object not loaded")
        body_index = self._model.body_name2id("object")
        num_joints = self._model.body_jntnum[body_index]
        start_index = self._model.body_jntadr[body_index]
        dof_index = 0
        for i in range(num_joints):
            joint_name = self._model.joint_id2name(start_index + i)
            joint_type = self._model.jnt_type[start_index + i]
            # Free joint
            if joint_type == 0:
                self._sim.data.set_joint_qpos(joint_name, pose[dof_index : dof_index + 7])
                self._sim.data.set_joint_qvel(joint_name, [0.0] * 6)
                dof_index += 6
            # Ball joint
            elif joint_type == 1:
                self._sim.data.set_joint_qpos(joint_name, pose[dof_index : dof_index + 4])
                self._sim.data.set_joint_qvel(joint_name, [0.0] * 4)
                dof_index += 3
            # Slide jointjoints
            elif joint_type == 2:
                self._sim.data.set_joint_qpos(joint_name, pose[dof_index])
                self._sim.data.set_joint_qvel(joint_name, 0.0)
                dof_index += 1
            # Hinge joint
            elif joint_type == 3:
                self._sim.data.set_joint_qpos(joint_name, pose[dof_index])
                self._sim.data.set_joint_qvel(joint_name, 0.0)
                dof_index += 1
        self._sim_data_dirty = True

    def _get_object_pos(self) -> np.ndarray:
        if self._sim_has_object() is False:
            raise RuntimeError("object not loaded")
        body_index = self._model.sensor_name2id("object_pos")
        index = self._model.sensor_adr[body_index]
        return self._sim.data.sensordata[index : index + 3].copy()

    def _get_object_quat(self) -> np.ndarray:
        if self._sim_has_object() is False:
            raise RuntimeError("object not loaded")
        body_index = self._model.sensor_name2id("object_quat")
        index = self._model.sensor_adr[body_index]
        quat = self._sim.data.sensordata[index : index + 4].copy()
        if quat[0] < 0:
            quat = -quat
        return quat

    def _get_object_angvel(self) -> np.ndarray:
        if self._sim_has_object() is False:
            raise RuntimeError("object not loaded")
        body_index = self._model.sensor_name2id("object_angvel")
        index = self._model.sensor_adr[body_index]
        return self._sim.data.sensordata[index : index + 3]

    # Trajplanner
    def _setup_trajplanners(self, trajplanner_params):
        logger.info("Setting up joint trajectory planners")
        frequency = trajplanner_params["frequency"]

        # Create trajectory planner class, one for each joint.
        self._trajplanners = []
        for i in range(self._num_hand_joints + self._num_arm_joints):
            self._trajplanners.append(TrajPlanner(**trajplanner_params))
        hand_joint_ctrl = self._get_hand_joint_ctrl()
        arm_joint_ctrl = self._get_arm_joint_ctrl()
        joint_ctrl = np.concatenate((hand_joint_ctrl, arm_joint_ctrl))
        # Set initial and setpoint positions for each trajectory planner
        # All trajectory planners are initialized to track the joint ctrl
        for i in range(self._num_hand_joints + self._num_arm_joints):
            self._trajplanners[i].plan(initial=joint_ctrl[i], setpoint=joint_ctrl[i])
        self._trajplanner_active = True
        self.register_callback(self._trajplanner_callback, frequency)

    def _trajplanner_callback(self):
        if self._trajplanner_active:
            ctrl = np.zeros(self._num_hand_joints + self._num_arm_joints)
            for i in range(self._num_hand_joints + self._num_arm_joints):
                trajplanner = self._trajplanners[i]
                trajplanner.step()
                ctrl[i] = trajplanner.get_control()
            self.set_ctrl(ctrl)

    def _set_setpoint(self, setpoint) -> None:
        """Set trajplanner setpoint"""
        # Current joint ctrl used as the initial values for planning setpoint trajectory
        hand_joint_ctrl = self._get_hand_joint_ctrl()
        arm_joint_ctrl = self._get_arm_joint_ctrl()
        joint_ctrl = np.concatenate((hand_joint_ctrl, arm_joint_ctrl))
        for trajplanner, initial, setpoint_ in zip(self._trajplanners, joint_ctrl, setpoint):
            trajplanner.plan(initial, setpoint_)

    def _get_setpoint(self):
        """Return setpoint of the joint trajectory planners"""
        setpoint = np.zeros(self._num_hand_joints + self._num_arm_joints)
        for i, trajplanner in enumerate(self._trajplanners):
            setpoint[i] = trajplanner.get_setpoint()
        return setpoint

    def _update_setpoint(self, dq):
        """Increment setpoint by dq"""
        # Current joint ctrl used as the initial values for planning setpoint trajectory
        hand_joint_ctrl = self._get_hand_joint_ctrl()
        arm_joint_ctrl = self._get_arm_joint_ctrl()
        joint_ctrl = np.concatenate((hand_joint_ctrl, arm_joint_ctrl))
        
        hand_joint_ctrl_range_low, hand_joint_ctrl_range_high = self._get_hand_joint_ctrl_range()
        arm_joint_ctrl_range_low, arm_joint_ctrl_range_high = self._get_arm_joint_ctrl_range()
        
        joint_ctrl_range_low = np.concatenate((hand_joint_ctrl_range_low, arm_joint_ctrl_range_low))
        joint_ctrl_range_high = np.concatenate((hand_joint_ctrl_range_high, arm_joint_ctrl_range_high))

        for trajplanner, initial, ctrl_low, ctrl_high, dq_ in zip(self._trajplanners, joint_ctrl, joint_ctrl_range_low, joint_ctrl_range_high, dq):
            setpoint = trajplanner.get_setpoint()
            new_setpoint = np.clip(setpoint + dq_, ctrl_low, ctrl_high)
            trajplanner.plan(initial, new_setpoint)

    def disable_trajplanner(self):
        self._trajplanner_active = False

    def enable_trajplanner(self):
        joint_ctrl = self._get_hand_joint_ctrl()
        for i in range(self._num_hand_joints):
            self._trajplanners[i].set_initial(joint_ctrl[i])
            self._trajplanners[i].set_setpoint(joint_ctrl[i])
        self._trajplanner_active = True

    def compute_ik_hand(self, action, roll=False):
        """ computes ik for the hand """
        # get link lengths
        link_lengths = self._link_lengths
        l_1, l_2, l_3 = link_lengths["finger1"]
        x_e, y_e, z_e = action[:, 0], action[:, 1], action[:, 2]
        l_prime = np.sqrt(x_e**2 + y_e**2 + (z_e - l_1)**2)
        beta = np.arctan2(z_e - l_1, np.sqrt(x_e**2 + y_e**2))
        alpha = np.arccos((l_2**2 + l_prime**2 - l_3**2)/(2*l_2*l_prime))
        a = l_prime * np.sin(alpha)

        theta2 = np.pi/2 - beta - alpha
        theta3 = np.arcsin(a/l_3)

        theta1 = np.arctan2(y_e, x_e) if roll else np.zeros_like(theta2)

        # ee_pos_cycle_check = self.compute_fk_hand(np.arctan2(y_e, x_e), theta2, theta3)
        # finger = 0

        current_joint_pos = self.hand_joint_pos.reshape(-1,3)
        np.set_printoptions(suppress=True)
        theta1_curr, theta2_curr, theta3_curr = current_joint_pos[:, 0], current_joint_pos[:, 1], current_joint_pos[:, 2]
        # print("current pos", self.compute_fk_hand(theta1_curr, theta2_curr, theta3_curr)[finger])
        action = np.array([theta1, theta2, theta3]).T
        # collapse the from 5x3 to 15
        action = action.flatten()
        return action
    
    def compute_fk_hand(self, theta_1, theta_2, theta_3):
        """ computes fk for the hand """
        l1, l2, l3 = self._link_lengths["finger1"]
        z_e = l1 + l2*np.cos(theta_2) + l3*np.cos(theta_2 + theta_3)
        projected_len = l2*np.sin(theta_2) + l3*np.sin(theta_2 + theta_3)
        x_e = projected_len*np.sin(theta_1)
        y_e = projected_len*np.cos(theta_1)
        return np.array([x_e, y_e, z_e]).T

    def compute_ik_arm(self, desired_ee_pose, current_arm_joint_pos):
        """ computes ik for the arm """
        padded_arm_joint_pos = np.pad(current_arm_joint_pos, (1, 1), 'constant', constant_values=(0, 0))                            
        arm_action = self.ur5_kinematics.inverse_kinematics_frame(desired_ee_pose, padded_arm_joint_pos, orientation_mode='all')
        masked_arm_action = arm_action[self.ur5_kinematics.active_links_mask]
        return masked_arm_action

    def check_ik_pos_hand(self, pos):
        """ checks if the position is within the workspace of the hand """
        link_lengths = self._link_lengths
        l_1, l_2, l_3 = link_lengths["finger1"]
        x_e, y_e, z_e = pos[:, 0], pos[:, 1], pos[:, 2]

        # clamp z from below to make it >= l_1
        z_e = np.clip(z_e, l_1, None)

        pos = np.array([x_e, y_e, z_e]).T
        valid = np.all(x_e**2 + y_e**2 + (z_e - l_1)**2 < (l_2 + l_3)**2)         

        # if not valid, project the point to the workspace
        if not valid:
            invalid_fingers = np.where(x_e**2 + y_e**2 + (z_e - l_1)**2 >= (l_2 + l_3)**2)[0]
            for i in invalid_fingers:
                centered_pos_i = pos[i] - np.array([0, 0, l_1])
                norm_centered_pos_i = np.linalg.norm(centered_pos_i)
                scale_i = 0.99 * (l_2 + l_3)/norm_centered_pos_i
                pos[i] = centered_pos_i*scale_i + np.array([0, 0, l_1])
                x_e, y_e, z_e = pos[i, 0], pos[i, 1], pos[i, 2]
                assert x_e**2 + y_e**2 + (z_e - l_1)**2 < (l_2 + l_3)**2

        x_e, y_e, z_e = pos[:, 0], pos[:, 1], pos[:, 2]
        assert np.all(x_e**2 + y_e**2 + (z_e - l_1)**2 < (l_2 + l_3)**2)


        l_prime = np.sqrt(x_e**2 + y_e**2 + (z_e - l_1)**2)   
        cos_alpha = (l_2**2 + l_prime**2 - l_3**2)/(2*l_2*l_prime)
        valid = np.all(cos_alpha >= -1) and np.all(cos_alpha <= 1)

        # typically caused by finger covering its own base --> y coord set to about 0
        if not valid:
            print("Invalid cos_alpha")
            invalid_fingers = np.where(np.logical_or(cos_alpha < -1, cos_alpha > 1))[0]
            for i in invalid_fingers:
                pos[i, 1] = np.sqrt(l_2**2 + l_3**2 - (pos[i,2]-l_1)**2)
                
        return pos

    def check_ik_pos_arm(self, pose):
        """Clamp ur5 ee position to avoid collision with table"""
        # run fk to get ee position
        ee_pos = pose[:3, 3]
        # table height
        half_table_height = self.table_height/2
        # link lengths
        link_lengths = self.get_link_lengths('finger1')
        l1, l2, l3 = link_lengths
        plate_depth = 0.0
        finger_reach = l1 + l2 + l3 + plate_depth
        # wrist workspace
        if ee_pos[2] - finger_reach < half_table_height:
            ee_pos[2] = half_table_height + l1 + l2 + l3
            pose[:3, 3] = ee_pos
        return pose

    # Contacts
    def _setup_tactile_sensor(self, tactile_sensor_params):
        logger.info("Setting up tactile sensor")
        self._tactile_sensor = TactileSensor(
            self._model,
            self._sim.data,
            self._get_ftip_links(),
            **tactile_sensor_params,
        )
        # List of ftip contacts returned by the tactile sensor
        self._ftip_contacts = []
        self.register_callback(self._tactile_sensor_callback, tactile_sensor_params["frequency"])

    def _tactile_sensor_callback(self):
        self._ftip_contacts = self._tactile_sensor.sense_contacts(add_noise=True)

    def _get_ftip_contacts(self):
        return self._ftip_contacts

    def _sense_ftip_contacts(self, add_noise=False):
        """
        Returns the contacts reported by the tactile sensor.

        The contacts reported by the tactile sensor are in finger frame.
        In this method, they are converted to global frame using
        forward kinematics. Thus propogating errors in kinematic model
        such as joint references, joint offsets to errors in contact
        position, normals and force vectors.
        """
        contacts = self._tactile_sensor.sense_contacts(add_noise)
        ftip_poses = self._get_ftip_poses()
        for contact in contacts:
            finger = contact["link"].replace("_distal", "")
            ftip_pos, ftip_mat = ftip_poses[finger]
            contact["position"] = np.dot(ftip_mat, contact["position"]) + ftip_pos
            contact["normal"] = np.dot(ftip_mat, contact["normal"])
            contact["force"] = np.dot(ftip_mat, contact["force"])
            contact["frame"] = ftip_mat
        return contacts

    def _get_ftip_contacts(self):
        return self._ftip_contacts

    def _get_ncontacts(self):
        return self._sim.data.ncon

    # Seeding
    def seed(self, seed):
        seed = seeding.hash_seed(seed)
        seeds = []
        for name in ["hand_joint_pos"]:
            seeds.append(seed)
            self._np_random[name], seed = seeding.np_random(seed)
        seeds.extend(self._tactile_sensor.seed(seed))
        return seeds

    # Pickling
    def __getstate__(self) -> Dict:
        state = super().__getstate__()
        state.update(
            {
                "default_hand_joint_pos": self._default_hand_joint_pos,
                "default_object_pose": self._default_object_pose,
                "simulation_settling_time": self._simulation_settling_time,
                "trajplanner_params": self._trajplanner_params,
                "tactile_sensor_params": self._tactile_sensor_params,
                "hand_joint_position_noise": self._hand_joint_position_noise,
            }
        )
        return state

    def close(self):
        # Hack to avoid huge memory leak
        # TODO: Re-implement to avoid circular references
        del self._tactile_sensor._model
        del self._tactile_sensor._data
        del self._tactile_sensor
        super().close()