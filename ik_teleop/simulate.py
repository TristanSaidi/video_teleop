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

from sim.hand import Hand

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
    def __init__(self, record_demo, hide_window, cfg = None, rotation_angle = 0, enable_moving_average = True, cache_file = None, calibration_duration = 100):
        self.record_demo = record_demo
        self.display_window= not hide_window

        # Getting the configurations
        if cfg is None:
            initialize(config_path = "./parameters/")
            self.cfg = compose(config_name = "teleop")
        else:
            self.cfg = cfg


        # self.robot = URDF.get_ur5(IHM_ROOT)
        # self.kinematics = Kinematics(self.robot)
        # self.controller = CartesianController(self.robot, self.kinematics)

        model_xml = Model3D5FH_UR5_PG.toxml(object="cube")
        self.ee_chain = chain.Chain.from_urdf_file(IHM_ROOT + "/model/urdf/ur5e.xml")
        # set first and last active links mask to False
        self.ee_chain.active_links_mask[0] = False # base_link
        self.ee_chain.active_links_mask[-1] = False # ee_link

        self.default_hand_joint_pos = np.array([0, 0.0, 0.0] * 5)
        self.default_arm_joint_pos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

        # print forward kinematics of default arm joint pos
        ee_pose = self.ee_chain.forward_kinematics(np.pad(self.default_arm_joint_pos, (1, 1), 'constant', constant_values=(0, 0)))
        print(f'ee pose: {ee_pose}')

        self.default_object_pose = np.array([0.5, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])

        
        # Ur5 mapping params
        # self.arm_workspace_center = np.array([0.10914794, -0.48689917, 0.43185934])
        self.arm_workspace_center = np.array([0.0, -0.5, 0.40])

        self.arm_workspace_width = np.array([0.25, 0.15, 0.15])


        self.sim = Hand(
            model=model_xml,
            default_hand_joint_pos=self.default_hand_joint_pos,
            default_arm_joint_pos=self.default_arm_joint_pos,
            default_object_pose=self.default_object_pose,
        )
        self.env = IHMEnv(
            sim=self.sim,
            max_episode_length=500,
            max_dq=0.05,
            discrete_action=False,
            randomize_initial_state=False,
        )
        
        self.viewer = self.env.sim._get_viewer("human")
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
            print(string)
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
            action = np.array([[0, 0, (l1+l2+l3)]]*5)

            # for calibration
            fingertip_pos_history = [] 
            wrist_pos_history = []
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
                # Create all the bounds in the image
                image = camera.create_contours(
                    image, 
                    self.cfg.mediapipe.contours.wrist_circle, 
                    self.cfg.mediapipe.contours.pinky_knuckle_bounds, 
                    self.cfg.mediapipe.contours.thumb_tip_bounds,
                    self.cfg.mediapipe.contours.thickness
                )

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

                        ## TODO: turn this into a function with edge checking
                        wrist_pos_z = depth[wrist_xy[1], wrist_xy[0]]
                        thumb_pos_z = depth[thumb_xy[1], thumb_xy[0]]
                        index_pos_z = depth[index_xy[1], index_xy[0]]
                        middle_pos_z = depth[middle_xy[1], middle_xy[0]]
                        ring_pos_z = depth[ring_xy[1], ring_xy[0]]
                        pinky_pos_z = depth[pinky_xy[1], pinky_xy[0]]
                        thumb_tip_pos_z = depth[finger_tip_positions['thumb'][1], finger_tip_positions['thumb'][0]]
                        index_tip_pos_z = depth[finger_tip_positions['index'][1], finger_tip_positions['index'][0]]
                        middle_tip_pos_z = depth[finger_tip_positions['middle'][1], finger_tip_positions['middle'][0]]
                        ring_tip_pos_z = depth[finger_tip_positions['ring'][1], finger_tip_positions['ring'][0]]
                        pinky_tip_pos_z = depth[finger_tip_positions['pinky'][1], finger_tip_positions['pinky'][0]]

                        # get xyz
                        wrist_position = [wrist_xy[0], wrist_xy[1], wrist_pos_z]
                        thumb_knuckle_position = [thumb_xy[0], thumb_xy[1], thumb_pos_z]
                        index_knuckle_position = [index_xy[0], index_xy[1], index_pos_z]
                        middle_knuckle_position = [middle_xy[0], middle_xy[1], middle_pos_z]
                        ring_knuckle_position = [ring_xy[0], ring_xy[1], ring_pos_z]
                        pinky_knuckle_position = [pinky_xy[0], pinky_xy[1], pinky_pos_z]
                        thumb_tip_position = [finger_tip_positions['thumb'][0], finger_tip_positions['thumb'][1], thumb_tip_pos_z]
                        index_tip_position = [finger_tip_positions['index'][0], finger_tip_positions['index'][1], index_tip_pos_z]
                        middle_tip_position = [finger_tip_positions['middle'][0], finger_tip_positions['middle'][1], middle_tip_pos_z]
                        ring_tip_position = [finger_tip_positions['ring'][0], finger_tip_positions['ring'][1], ring_tip_pos_z]
                        pinky_tip_position = [finger_tip_positions['pinky'][0], finger_tip_positions['pinky'][1], pinky_tip_pos_z]

                        finger_tip_positions = {
                            'thumb': thumb_tip_position,
                            'index': index_tip_position,
                            'middle': middle_tip_position,
                            'ring': ring_tip_position,
                            'pinky': pinky_tip_position
                        }

                        finger_base_positions = {
                            'thumb': thumb_knuckle_position,
                            'index': index_knuckle_position,
                            'middle': middle_knuckle_position,
                            'ring': ring_knuckle_position,
                            'pinky': pinky_knuckle_position
                        }

                        hand_state_dict = {
                            "wrist": wrist_position,
                            "fingertips": finger_tip_positions,
                            "finger_bases": finger_base_positions
                        }

                        # get the z axis of the fingers (vector from palm to finger base)
                        finger_frames = self.compute_finger_frames_palm_down(hand_state_dict)
                        
                        # get the finger tip positions in the finger frames
                        finger_tip_positions_finger_frame = self.get_ftip_pos_finger_frame(finger_frames, finger_tip_positions, finger_base_positions)

                        # collect data for calibration
                        if self.timestep < self.calibration_duration and self.calibrate:
                            # turn into a [5, 3] array
                            finger_tip_positions_finger_frame = np.array([finger_tip_positions_finger_frame[finger] for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']])
                            fingertip_pos_history.append(finger_tip_positions_finger_frame)
                            wrist_pos_history.append(wrist_position)
                            print(f'calibrating... {self.timestep}/{self.calibration_duration}')

                        # compute empirical ranges at the end of calibration (with filtering)
                        elif self.timestep == self.calibration_duration and self.calibrate:
                            fingertip_pos_history = np.array(fingertip_pos_history)
                            
                            wrist_pos_history = np.array(wrist_pos_history).astype(float)
                            filtered_wrist_pos_history = scipy.signal.medfilt(wrist_pos_history, (5, 1))
                            
                            # median filter along the first (time) axis to remove outliers
                            filtered_fingertip_pos_history = scipy.signal.medfilt(fingertip_pos_history, (5, 1, 1))

                            # for each (finger, axis) pair, store the max and min in a dict
                            finger_empirical_ranges = {}
                            for i, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
                                finger_max = np.max(filtered_fingertip_pos_history[:, i, :], axis=0)
                                finger_min = np.min(filtered_fingertip_pos_history[:, i, :], axis=0)
                                finger_range = [(min, max) for min, max in zip(finger_min, finger_max)]
                                finger_empirical_ranges[finger] = finger_range
                            wrist_max = np.max(filtered_wrist_pos_history, axis=0)
                            wrist_min = np.min(filtered_wrist_pos_history, axis=0)
                            wrist_range = [(min, max) for min, max in zip(wrist_min, wrist_max)]
                            finger_empirical_ranges['wrist'] = wrist_range
                            if self.write_cache:
                                with open(os.path.join('cache', self.cache_file), 'wb') as f:
                                    pickle.dump(finger_empirical_ranges, f)
                        else:

                            action, wrist_position = self.compute_scaled_action(
                                finger_tip_positions_finger_frame, 
                                wrist_position, 
                                finger_empirical_ranges, 
                                l1, 
                                l2, 
                                l3, 
                                self.arm_workspace_center, 
                                self.arm_workspace_width
                            )
                            ee_rot = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]) # constant wrist orientation for now

                            # clamp wrist position to workspace
                            wrist_position = np.clip(wrist_position, self.arm_workspace_center - self.arm_workspace_width, self.arm_workspace_center + self.arm_workspace_width)

                            desired_ee_pose = np.eye(4)
                            desired_ee_pose[:3, 3] = wrist_position
                            desired_ee_pose[:3, :3] = ee_rot

                            # inverse kinematics
                            current_arm_joint_pos = self.sim.arm_joint_pos.copy()
                            padded_arm_joint_pos = np.pad(current_arm_joint_pos, (1, 1), 'constant', constant_values=(0, 0))                            

                            arm_action = self.ee_chain.inverse_kinematics_frame(desired_ee_pose, padded_arm_joint_pos, orientation_mode='all')
                            masked_arm_action = arm_action[self.ee_chain.active_links_mask]


                            # cycle consistency - forward kinematics
                            ee_pose = self.ee_chain.forward_kinematics(arm_action)
                            ee_pos = ee_pose[:3, 3]

                            # if inconsistent, use previous action
                            wrist_displacement = np.linalg.norm(wrist_position - ee_pos)
                            if wrist_displacement > 1e-9:
                                print(f'cycle consistency error: {wrist_displacement}')
                                masked_arm_action = prev_action_dict["arm"]

                            # step
                            action_dict = {
                                "hand": action,
                                "arm": masked_arm_action
                            }
                            # fetch current ur5 joint angles
                            obs, _, done, info = self.env.step(action_dict)

                            # save previous action
                            prev_action_dict = action_dict
                            
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


    def compute_scaled_action(self, finger_tip_positions_finger_frame, wrist_position, finger_empirical_ranges, l1, l2, l3, wrist_worspace_center, wrist_workspace_width):
        # scale the finger tip position based on the empirical range
        # final end effector position (action) should be in the hemispherical workspace of the finger
        ftip_pos_array = []
        for finger in finger_tip_positions_finger_frame.keys():
            ftip_position = finger_tip_positions_finger_frame[finger]
            # scale the finger tip position based on the empirical range
            empirical_ranges = finger_empirical_ranges[finger]
            empirical_range_x, empirical_range_y, empirical_range_z = empirical_ranges
            # x
            mean = (empirical_range_x[0] + empirical_range_x[1]) / 2
            width = (empirical_range_x[1] - empirical_range_x[0]) / 2
            x = (ftip_position[0] - mean) * (l2 + l3) / width
            # y
            mean = (empirical_range_y[0] + empirical_range_y[1]) / 2
            width = (empirical_range_y[1] - empirical_range_y[0]) / 2
            y = (ftip_position[1] - mean) * (l2 + l3) / width
            # z
            mean = (empirical_range_z[0] + empirical_range_z[1]) / 2
            width = (empirical_range_z[1] - empirical_range_z[0]) / 2
            z = ((ftip_position[2] - mean) * (l2 + l3) / width) + l1
            ftip_pos_array.append([x, y, z])
        # wrist position
        empirical_ranges = finger_empirical_ranges['wrist']
        empirical_range_x, empirical_range_y, empirical_range_z = empirical_ranges
        # x
        mean = (empirical_range_x[0] + empirical_range_x[1]) / 2
        width = (empirical_range_x[1] - empirical_range_x[0]) / 2
        # min wrist x position should map to highest x value
        x = (wrist_position[0] - mean) * -1 * wrist_workspace_width[0] / width + wrist_worspace_center[0]
        # y
        mean = (empirical_range_y[0] + empirical_range_y[1]) / 2
        width = (empirical_range_y[1] - empirical_range_y[0]) / 2
        y = (wrist_position[1] - mean) * wrist_workspace_width[1] / width + wrist_worspace_center[1]
        # z
        mean = (empirical_range_z[0] + empirical_range_z[1]) / 2
        width = (empirical_range_z[1] - empirical_range_z[0]) / 2

        # min wrist z position should map to highest z value
        z = (wrist_position[2] - mean) * -1 * wrist_workspace_width[2] / width + wrist_worspace_center[2]
        
        ftip_pos_array = np.array(ftip_pos_array)
        wrist_pos = np.array([x, y, z])
        # project the end effector position onto the hemispherical workspace of the finger
        ftip_pos_array = self.env.sim.check_ik_pos(ftip_pos_array)
        return ftip_pos_array, wrist_pos

    def compute_finger_frames_palm_up(self, hand_state_dict):
        # computes the normalized frame the fingers 
        # (z = vector from palm to finger base)
        wrist_position = hand_state_dict['wrist']
        finger_base_positions = hand_state_dict['finger_bases']
        finger_z_axes = {
            'thumb': np.array(finger_base_positions['thumb']) - np.array(wrist_position),
            'index': np.array(finger_base_positions['index']) - np.array(wrist_position),
            'middle': np.array(finger_base_positions['middle']) - np.array(wrist_position),
            'ring': np.array(finger_base_positions['ring']) - np.array(wrist_position),
            'pinky': np.array(finger_base_positions['pinky']) - np.array(wrist_position)
        }
        # normalize
        for finger in finger_z_axes:
            finger_z_axes[finger] = finger_z_axes[finger] / np.linalg.norm(finger_z_axes[finger])

        # compute y axis by taking the cross product of the z of consecutive fingers
        finger_y_axes = {
            'thumb': np.cross(finger_z_axes['index'], finger_z_axes['thumb']),
            'index': np.cross(finger_z_axes['middle'], finger_z_axes['index']),
            'middle': np.cross(finger_z_axes['ring'], finger_z_axes['middle']),
            'ring': np.cross(finger_z_axes['pinky'], finger_z_axes['ring']),
            'pinky': np.cross(finger_z_axes['pinky'], finger_z_axes['ring'])
        }
        # normalize
        for finger in finger_y_axes:
            finger_y_axes[finger] = finger_y_axes[finger] / np.linalg.norm(finger_y_axes[finger])
        
        # compute x axis by taking the cross product of the y axis and the z axis
        finger_x_axes = {
            'thumb': np.cross(finger_y_axes['thumb'], finger_z_axes['thumb']),
            'index': np.cross(finger_y_axes['index'], finger_z_axes['index']),
            'middle': np.cross(finger_y_axes['middle'], finger_z_axes['middle']),
            'ring': np.cross(finger_y_axes['ring'], finger_z_axes['ring']),
            'pinky': np.cross(finger_y_axes['pinky'], finger_z_axes['pinky'])
        }
        # normalize
        for finger in finger_x_axes:
            finger_x_axes[finger] = finger_x_axes[finger] / np.linalg.norm(finger_x_axes[finger])

        # construct transforms
        finger_frames = {
            'thumb': np.array([finger_x_axes['thumb'], finger_y_axes['thumb'], finger_z_axes['thumb']]),
            'index': np.array([finger_x_axes['index'], finger_y_axes['index'], finger_z_axes['index']]),
            'middle': np.array([finger_x_axes['middle'], finger_y_axes['middle'], finger_z_axes['middle']]),
            'ring': np.array([finger_x_axes['ring'], finger_y_axes['ring'], finger_z_axes['ring']]),
            'pinky': np.array([finger_x_axes['pinky'], finger_y_axes['pinky'], finger_z_axes['pinky']])
        }

        return finger_frames
    

    def compute_finger_frames_palm_down(self, hand_state_dict):
        # computes the normalized frame the fingers 
        # (z = vector from palm to finger base)
        wrist_position = hand_state_dict['wrist']
        finger_base_positions = hand_state_dict['finger_bases']
        finger_z_axes = {
            'thumb': np.array(finger_base_positions['thumb']) - np.array(wrist_position),
            'index': np.array(finger_base_positions['index']) - np.array(wrist_position),
            'middle': np.array(finger_base_positions['middle']) - np.array(wrist_position),
            'ring': np.array(finger_base_positions['ring']) - np.array(wrist_position),
            'pinky': np.array(finger_base_positions['pinky']) - np.array(wrist_position)
        }
        # normalize
        for finger in finger_z_axes:
            finger_z_axes[finger] = finger_z_axes[finger] / np.linalg.norm(finger_z_axes[finger])

        # compute y axis by taking the cross product of the z of consecutive fingers
        finger_y_axes = {
            'thumb': np.cross(finger_z_axes['thumb'], finger_z_axes['index']),
            'index': np.cross(finger_z_axes['index'], finger_z_axes['middle']),
            'middle': np.cross(finger_z_axes['middle'], finger_z_axes['ring']),
            'ring': np.cross(finger_z_axes['ring'], finger_z_axes['pinky']),
            'pinky': np.cross(finger_z_axes['ring'], finger_z_axes['pinky'])
        }
        # normalize
        for finger in finger_y_axes:
            finger_y_axes[finger] = finger_y_axes[finger] / np.linalg.norm(finger_y_axes[finger])
        
        # compute x axis by taking the cross product of the y axis and the z axis
        finger_x_axes = {
            'thumb': np.cross(finger_y_axes['thumb'], finger_z_axes['thumb']),
            'index': np.cross(finger_y_axes['index'], finger_z_axes['index']),
            'middle': np.cross(finger_y_axes['middle'], finger_z_axes['middle']),
            'ring': np.cross(finger_y_axes['ring'], finger_z_axes['ring']),
            'pinky': np.cross(finger_y_axes['pinky'], finger_z_axes['pinky'])
        }
        # normalize
        for finger in finger_x_axes:
            finger_x_axes[finger] = finger_x_axes[finger] / np.linalg.norm(finger_x_axes[finger])

        # construct transforms
        finger_frames = {
            'thumb': np.array([finger_x_axes['thumb'], finger_y_axes['thumb'], finger_z_axes['thumb']]),
            'index': np.array([finger_x_axes['index'], finger_y_axes['index'], finger_z_axes['index']]),
            'middle': np.array([finger_x_axes['middle'], finger_y_axes['middle'], finger_z_axes['middle']]),
            'ring': np.array([finger_x_axes['ring'], finger_y_axes['ring'], finger_z_axes['ring']]),
            'pinky': np.array([finger_x_axes['pinky'], finger_y_axes['pinky'], finger_z_axes['pinky']])
        }

        return finger_frames


    def get_ftip_pos_finger_frame(self, finger_frames, finger_tip_positions, finger_base_positions):
        # get the finger tip positions in the finger frames
        assert finger_frames.keys() == finger_tip_positions.keys() == finger_base_positions.keys()
        keys = finger_frames.keys()
        # compute ftip-base vectors
        ftip_base_vectors = {}
        for finger in keys:
            ftip_base_vectors[finger] = np.array(finger_tip_positions[finger]) - np.array(finger_base_positions[finger])
        # project ftip-base vectors onto finger frames
        ftip_pos_finger_frame = {}
        for finger in keys:
            ftip_pos_finger_frame[finger] = finger_frames[finger] @ ftip_base_vectors[finger]
        return ftip_pos_finger_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use teleop to operate Mujoco sim')
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--headless', type=bool, default=False)
    parser.add_argument('--cache_file', default=None)
    parser.add_argument('--calibration_duration', default=100, type=int)
    args = parser.parse_args()

    teleop = TeleOpSim(args.record, args.headless, rotation_angle = 0, cache_file = args.cache_file, calibration_duration = args.calibration_duration)
    # embed()
    try:
        teleop.hand_movement_processor()
    except KeyboardInterrupt:
        print ('Interrupted')
        if(teleop.record_demo):
            teleop.finish_recording()
            print(teleop.demo_dir)
        sys.exit(0)
