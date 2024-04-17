# Standard imports
import os
import sys
import numpy as np

IHM_ROOT = '/home/tristan/Research/shared_aut/DIME-IK-TeleOp/ihmgym'
IKFP_ROOT = '/home/tristan/Research/ikfastpy'
sys.path.append(IHM_ROOT)
sys.path.append(IKFP_ROOT)
from envs.ihm_env import IHMEnv
from model.model3d5fh import Model3D5FH
from model.model3d5fh_ur5e import Model3D5FH_UR5_PG
from model.model3d4fh_ur5e import Model3D4FH_UR5_PG

from sim.hand import Hand
from finger_frames.finger_frames import *

import ikfastpy

import scipy
import matplotlib.pyplot as plt 
# Parameter management imports
from hydra import initialize, compose

# Image based imports
import cv2
import mediapipe
import pyrealsense2 as rs

from sensor_msgs.msg import JointState

from datetime import datetime

# Other utility imports
import utils.camera as camera
import utils.joint_handling as joint_handlers
from utils.transformations import perform_persperctive_transformation

# Other miscellaneous imports
from copy import deepcopy as copy

import argparse
import pickle
import tf
# Debugging imports
from IPython import embed
from ikpy import chain

class TeleOpSim(object):
    def __init__(self, record_demo, hide_window, cfg = None, num_fingers=4, rotation_angle = 0, enable_moving_average = True, cache_file = None, calibration_duration = 100):
        self.record_demo = record_demo
        self.display_window= not hide_window

        # Getting the configurations
        if cfg is None:
            initialize(config_path = "./parameters/")
            self.cfg = compose(config_name = "teleop")
        else:
            self.cfg = cfg

        self.table_height = 0.1
        Kp = 100

        self.num_fingers = num_fingers
        kp_prox, kp_middle, kp_dist = np.array([Kp]*self.num_fingers), np.array([Kp]*self.num_fingers), np.array([Kp]*self.num_fingers)
        
        model_class = Model3D5FH_UR5_PG if self.num_fingers == 5 else Model3D4FH_UR5_PG
        
        model_xml = model_class.toxml(
            object="cube", 
            table_height=self.table_height,
            kp_prox=kp_prox,
            kp_middle=kp_middle,
            kp_dist=kp_dist
        )
        self.ee_chain = chain.Chain.from_urdf_file(IHM_ROOT + "/model/urdf/ur5e.xml")
        # set first and last active links mask to False
        self.ee_chain.active_links_mask[0] = False # base_link
        self.ee_chain.active_links_mask[-1] = False # ee_link

        self.default_hand_joint_pos = np.array([0, 0.0, 0.0] * self.num_fingers)
        self.default_arm_joint_pos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708])
        self.default_ee_rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        # print forward kinematics of default arm joint pos
        ee_pose = self.ee_chain.forward_kinematics(np.pad(self.default_arm_joint_pos, (1, 1), 'constant', constant_values=(0, 0)))

        self.default_object_pose = np.array([0.5, -0.3, 0.25, 1.0, 0.0, 0.0, 0.0])
        
        # Ur5 mapping params
        # self.arm_workspace_center = np.array([0.10914794, -0.48689917, 0.43185934])
        self.arm_workspace_center = np.array([0.0, -0.5, 0.40])

        self.arm_workspace_width = np.array([0.5, 0.15, 0.15])


        self.sim = Hand(
            model=model_xml,
            ur5_kinematics=self.ee_chain,
            default_hand_joint_pos=self.default_hand_joint_pos,
            default_arm_joint_pos=self.default_arm_joint_pos,
            default_object_pose=self.default_object_pose,
        )
        
        max_dq_hand = [0.05] * self.num_fingers * 3
        max_dq_arm = [0.1] * 6
        max_dq = np.array(max_dq_hand + max_dq_arm)
        
        self.env = IHMEnv(
            sim=self.sim,
            max_episode_length=500,
            max_dq=max_dq,
            discrete_action=False,
            randomize_initial_state=False,
            num_fingers=self.num_fingers,
        )
        
        self.viewer = self.env.sim._get_viewer("human")
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -22.5
        initial_env = self.env.reset()
        
        # Creating a realsense pipeline
        # for cam in self.cfg.realsense.serial_numbers:
        self.pipeline, config = camera.create_realsense_rgb_depth_pipeline(self.cfg.realsense.serial_numbers[3], self.cfg.realsense.resolution, self.cfg.realsense.fps)
        self.pipeline.start()

        self.calibration_duration = calibration_duration
        self.load_cache = cache_file is not None and os.path.exists(os.path.join('cache', cache_file))
        self.write_cache = cache_file is not None and not os.path.exists(os.path.join('cache', cache_file))
        if cache_file is not None:
            string = f"Loading from {os.path.join('cache', cache_file)}" if self.load_cache else f"Writing to {os.path.join('cache', cache_file)}"
        self.cache_file = cache_file
        # perform calibration if we are writing to cache or if we are not loading from cache
        self.calibrate = self.write_cache or self.cache_file is None
        # Creating mediapipe objects
        self.mediapipe_drawing = mediapipe.solutions.drawing_utils
        self.mediapipe_hands = mediapipe.solutions.hands

        # Initializing a current joint state variable and creating a subscriber to get the current allegro joint angles
        self.current_joint_state = np.ones(16) * 0.2

        # Moving average arrays
        self.enable_moving_average = enable_moving_average
        if self.enable_moving_average is True:
            self.moving_average_queues = {
                'thumb': [],
                'index': [],
                'middle': [],
                'ring': []
            }

        # Storing the specified camera rotation
        self.rotation_angle = rotation_angle
        
        #Used for recording images and states
        if(not os.path.isdir('demos')):
            os.mkdir('demos')
        t= datetime.now()
        date_str = t.strftime('%b_%d_%H_%M')

        self.obs_freq = 1
        self.obs_ctr = 0
        self.demo_dir = os.path.join('demos',"demo_{}".format(date_str))
        if(self.record_demo and not os.path.isdir(self.demo_dir)):
            os.mkdir(self.demo_dir)
        self.vid_file = os.path.join(self.demo_dir, 'demo.mp4')
        self.unmrkd_file = os.path.join(self.demo_dir,'orig.mp4')
        self.pkl_file = os.path.join(self.demo_dir, 'd_{}.pickle'.format(date_str))
        
        self.demo_dict = {}
        self.current_state = None
        self.set_init_state = False
        self.unmrkd_images = []
        self.images = []
        self.current_states = [] 

    def add_im(self, image):
        self.images.append(image)

    def add_demo_entry(self, env, action, obs, reward, done, info):
        rentry = np.expand_dims(np.array(reward),0)
        if(self.demo_dict['rewards'] is None):
            self.demo_dict['rewards'] = rentry
        else:
            self.demo_dict['rewards'] = np.concatenate((self.demo_dict['rewards'],rentry),0)

        qvel = self.env.env.sim.data.qvel
        qvel = np.expand_dims(qvel, 0)

        qpos = obs
        qpos = np.expand_dims(qpos, 0)

        desired_goal = np.expand_dims(env.env.desired_orien,0)
        if (self.demo_dict['env_infos']['desired_orien'] is None):
            self.demo_dict['env_infos']['desired_orien'] = desired_goal
        else:
            self.demo_dict['env_infos']['env_infos'] = np.concatenate((self.demo_dict['env_infos']['desired_orien'], desired_goal),0)

        if (self.demo_dict['env_infos']['qvel'] is None):
            self.demo_dict['env_infos']['qvel'] = qvel
        else:
            self.demo_dict['env_infos']['qvel'] = np.concatenate((self.demo_dict['env_infos']['qvel'], qvel),0)

        if (self.demo_dict['env_infos']['qpos'] is None):
            self.demo_dict['env_infos']['qpos'] = qpos
        else:
            self.demo_dict['env_infos']['qpos'] = np.concatenate((self.demo_dict['env_infos']['qpos'], qpos),0)

        if (self.demo_dict['observations'] is None):
            self.demo_dict['observations'] = qpos
        else:
            self.demo_dict['observations'] = np.concatenate((self.demo_dict['observations'], qpos),0)


        action = np.expand_dims(action, 0)
        if (self.demo_dict['actions'] is None):
            self.demo_dict['actions'] = action
        else:
            self.demo_dict['actions'] = np.concatenate((self.demo_dict['actions'], action),0)

        
    def finish_recording(self):
        vid_writer = cv2.VideoWriter(self.vid_file,cv2.VideoWriter_fourcc(*'mp4v'), self.cfg.realsense.fps, self.cfg.realsense.resolution)
        for im in self.images:
            vid_writer.write(im)
        vid_writer.release

        uvid_writer = cv2.VideoWriter(self.unmrkd_file,cv2.VideoWriter_fourcc(*'mp4v'), self.cfg.realsense.fps, self.cfg.realsense.resolution)
        for im in self.unmrkd_images:
            uvid_writer.write(im)
        uvid_writer.release()

        file = open(self.pkl_file,'wb')
        pickle.dump(self.demo_dict, file)


    def hand_movement_processor(self):
        # Setting the mediapipe hand parameters
        with self.mediapipe_hands.Hands(
            max_num_hands = 1, # Limiting the number of hands detected in the image to 1
            min_detection_confidence = 0.75,
            min_tracking_confidence = 0.75) as hand:

            align_to = rs.stream.color
            align = rs.align(align_to)

            # initialize action
            l1, l2, l3 = self.sim.get_link_lengths('finger1')
            action = np.array([[0, 0, (l1+l2+l3)]]*self.num_fingers)

            # for calibration
            fingertip_pos_history = [] 
            finger_empirical_ranges = None
            
            # if we are loading from cache, load the empirical ranges
            if self.load_cache:
                with open(os.path.join('cache', self.cache_file), 'rb') as f:
                    finger_empirical_ranges = pickle.load(f)
                self.calibrate = False
            
            prev_action_dict = {
                "hand": action,
                "arm": self.default_arm_joint_pos
            }
            self.timestep = 0

            while True:
                # Getting the image to process
                # image = camera.getting_image_data(self.pipeline)
                # depth = camera.getting_depth_data(self.pipeline)

                frames = self.pipeline.wait_for_frames()
                
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                ### process color frames
                # Storing the frame data as an image
                image = np.asanyarray(color_frame.get_data())
                if image is None:
                    print('Ignoring empty camera frame!')
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                ### process depth frames
                depth = np.asanyarray(depth_frame.get_data())

                # Rotate image if needed
                if self.rotation_angle != 0:
                    image = camera.rotate_image(image, self.rotation_angle)
                

                # Getting the hand pose results out of the image
                image.flags.writeable = False
                estimate = hand.process(image)
                image.flags.writeable = True

                # Converting the image back from RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                raw_im = copy(image)

                # If there is a mediapipe hand estimate
                if estimate.multi_hand_landmarks is not None:  
                    # Getting the hand coordinate values for the only detected hand
                    hand_landmarks = estimate.multi_hand_landmarks[0]
                    # Embedding the hand drawind in the image
                    self.mediapipe_drawing.draw_landmarks(
                            image, hand_landmarks, self.mediapipe_hands.HAND_CONNECTIONS)

                    if self.current_joint_state is not None:
                        # Getting the mediapipe wrist and fingertip positions
                        # wrist_position, thumb_knuckle_position, index_knuckle_position, middle_knuckle_position, ring_knuckle_position, pinky_knuckle_position, finger_tip_positions = joint_handlers.get_joint_positions(hand_landmarks, self.cfg.realsense.resolution, self.cfg.mediapipe) 
                        wrist_xy, thumb_xy, index_xy, middle_xy, ring_xy, pinky_xy, finger_tip_positions = joint_handlers.get_joint_positions(hand_landmarks, self.cfg.realsense.resolution, self.cfg.mediapipe)
                        # get z pos from depth

                        xy_pos_dict = {
                            'wrist': wrist_xy,
                            'thumb': thumb_xy,
                            'index': index_xy,
                            'middle': middle_xy,
                            'ring': ring_xy,
                            'pinky': pinky_xy,
                            'finger_tip_positions': finger_tip_positions
                        }

                        xyz_pos_dict = joint_handlers.compute_hand_landmarks_xyz(
                            xy_pos_dict,
                            depth,
                            self.cfg.realsense.resolution
                        )

                        finger_tip_positions = {
                            'thumb': xyz_pos_dict['thumb_tip'],
                            'index': xyz_pos_dict['index_tip'],
                            'middle': xyz_pos_dict['middle_tip'],
                            'ring': xyz_pos_dict['ring_tip'],
                            'pinky': xyz_pos_dict['pinky_tip']
                        }

                        finger_base_positions = {
                            'thumb': xyz_pos_dict['thumb'],
                            'index': xyz_pos_dict['index'],
                            'middle': xyz_pos_dict['middle'],
                            'ring': xyz_pos_dict['ring'],
                            'pinky': xyz_pos_dict['pinky']
                        }

                        wrist_position = xyz_pos_dict['wrist']
                        joint_handlers.check_wrist_position(wrist_position, dict(self.cfg.wrist_bounds))

                        hand_state_dict = {
                            "wrist": wrist_position,
                            "fingertips": finger_tip_positions,
                            "finger_bases": finger_base_positions
                        }

                        # get the z axis of the fingers (vector from palm to finger base)
                        finger_frames = compute_finger_frames_palm_up(hand_state_dict)
                        
                        # get the finger tip positions in the finger frames
                        finger_tip_positions_finger_frame = get_ftip_pos_finger_frame(finger_frames, finger_tip_positions, finger_base_positions)

                        # collect data for calibration
                        if self.timestep < self.calibration_duration and self.calibrate:
                            # turn into a [5, 3] array
                            finger_tip_positions_finger_frame = np.array([finger_tip_positions_finger_frame[finger] for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']])
                            fingertip_pos_history.append(finger_tip_positions_finger_frame)
                            print(f'calibrating... {self.timestep}/{self.calibration_duration}')

                        # compute empirical ranges at the end of calibration (with filtering)
                        elif self.timestep == self.calibration_duration and self.calibrate:
                            fingertip_pos_history = np.array(fingertip_pos_history)                            
                            # median filter along the first (time) axis to remove outliers
                            filtered_fingertip_pos_history = scipy.signal.medfilt(fingertip_pos_history, (5, 1, 1))

                            # for each (finger, axis) pair, store the max and min in a dict
                            finger_empirical_ranges = {}
                            for i, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
                                finger_max = np.max(filtered_fingertip_pos_history[:, i, :], axis=0)
                                finger_min = np.min(filtered_fingertip_pos_history[:, i, :], axis=0)
                                finger_range = [(min, max) for min, max in zip(finger_min, finger_max)]
                                finger_empirical_ranges[finger] = finger_range
                            if self.write_cache:
                                with open(os.path.join('cache', self.cache_file), 'wb') as f:
                                    pickle.dump(finger_empirical_ranges, f)
                        else:
                            normalized_ftip_positions = []
                            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                                normalized_pos = compute_normalized_position(finger_tip_positions_finger_frame[finger], finger_empirical_ranges[finger])
                                normalized_ftip_positions.append(normalized_pos)
                            normalized_ftip_positions = np.array(normalized_ftip_positions)
                            normalized_wrist_position = compute_normalized_position(wrist_position, dict(self.cfg.wrist_bounds))
                            # compute action
                            action = np.zeros((self.num_fingers, 3))
                            workspace_center = [0.0, 0.0, l1 + (l2 + l3)/2] # y goes from 0 to l2+l3
                            workspace_width = [l2+l3, (l2+l3)/2, (l2+l3)/2]
                            
                            for i in range(self.num_fingers):
                                norm_ftip_i = normalized_ftip_positions[i, :]
                                action[i, :] = np.array(compute_action(norm_ftip_i, workspace_center, workspace_width))
                            infer_y = True
                            
                            if infer_y:
                                action[:, 1] = compute_y_from_z(action[:, 2], l1, l2, l3)
                                action[:, 0] = 0 # zero out x
 
                            wrist_position = np.array(
                                compute_action(
                                    normalized_wrist_position, 
                                    self.arm_workspace_center, 
                                    self.arm_workspace_width,
                                    invert_mask = [True, False, True] # invert x and z axes
                                )
                            )

                            # infer roll
                            yaw, roll = infer_yaw_roll(hand_state_dict)

                            ee_rot = self.default_ee_rot # constant wrist orientation for now
                            # apply roll
                            # ee_rot = np.dot(np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]]), ee_rot)
                            # apply yaw
                            # ee_rot = np.dot(np.array([[1, 0, 0], [0, np.cos(yaw), -np.sin(yaw)], [0, np.sin(yaw), np.cos(yaw)]]), ee_rot)
                            # clamp wrist position to workspace
                            wrist_position = np.clip(wrist_position, self.arm_workspace_center - self.arm_workspace_width, self.arm_workspace_center + self.arm_workspace_width)

                            desired_ee_pose = np.eye(4)
                            desired_ee_pose[:3, 3] = wrist_position
                            desired_ee_pose[:3, :3] = ee_rot

                            # step
                            action_dict = {
                                "hand": action,
                                "arm": desired_ee_pose,
                            }
                            # fetch current ur5 joint angles
                            obs, _, done, info = self.env.step(action_dict)
                        self.timestep += 1

                # Printing the image
                if(self.display_window):
                    cv2.imshow('Teleop - Mediapipe screen', image)
                    # cv2.imshow('Teleop - Raw screen', depth)
                    # self.viewer.add_marker(pos=action[0], size=[0.01, 0.01, 0.01], label='thumb', type=2)
                    # self.viewer.add_marker(pos=action[1], size=[0.01, 0.01, 0.01], label='index', type=2)
                    # self.viewer.add_marker(pos=action[2], size=[0.01, 0.01, 0.01], label='middle', type=2)
                    # self.viewer.add_marker(pos=action[3], size=[0.01, 0.01, 0.01], label='ring', type=2)
                    # self.viewer.add_marker(pos=action[4], size=[0.01, 0.01, 0.01], label='pinky', type=2)
                    self.viewer.render()

                # self.env.env.reset_model()
                # Condition to break the loop incase of keyboard interrupt
                if cv2.waitKey(30) & 0xFF == 27:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use teleop to operate Mujoco sim')
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--headless', type=bool, default=False)
    parser.add_argument('--cache_file', default=None)
    parser.add_argument('--num_fingers', default=4, type=int)
    parser.add_argument('--calibration_duration', default=100, type=int)
    args = parser.parse_args()

    teleop = TeleOpSim(args.record, args.headless, num_fingers=args.num_fingers, rotation_angle = 0, cache_file = args.cache_file, calibration_duration = args.calibration_duration)
    # embed()
    try:
        teleop.hand_movement_processor()
    except KeyboardInterrupt:
        print ('Interrupted')
        if(teleop.record_demo):
            teleop.finish_recording()
            print(teleop.demo_dir)
        sys.exit(0)
