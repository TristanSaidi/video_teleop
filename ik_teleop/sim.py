# Standard imports
import os
import sys
import numpy as np

# Parameter management imports
from hydra import initialize, compose

# Image based imports
import cv2
import mediapipe
import pyrealsense2 as rs

# Gym imports
from mjrl.utils.gym_env import GymEnv
import mj_allegro_envs

from sensor_msgs.msg import JointState

from datetime import datetime

# Allegro Inverse Kinematics based controller
from ik_teleop.ik_core.allegro_controller import AllegroIKController

# Other utility imports
import utils.camera as camera
import utils.joint_handling as joint_handlers
from utils.transformations import perform_persperctive_transformation

# Other miscellaneous imports
from copy import deepcopy as copy

import argparse
import pickle

# Debugging imports
from IPython import embed

class TeleOpSim (object):
    def __init__(self, record_demo, hide_window, cfg = None, rotation_angle = 0, enable_moving_average = True):
        self.record_demo = record_demo
        self.display_window= not hide_window

        # Getting the configurations
        if cfg is None:
            initialize(config_path = "./parameters/")
            self.cfg = compose(config_name = "teleop")
        else:
            self.cfg = cfg

        # self.env = gym.make('AllegroManipulateBlockRotateZPalm-v0')
        self.env = GymEnv('block-v3')
        initial_env = self.env.reset()

        #self.desired_angles = np.ones(16)*0.6
        # self.desired_angles = np.array([0.2, 0.17723984, 0.22907847, 0.2, 0.2, 0.17268343, 
        #                     0.30023316, 0.2, 0.28197248, 0.28114717, 1.41374226, 1.60438803,
        #                     0.37764562, 0.89932933, 1.06350173, 1.68177814])
        self.desired_angles = np.array([0.2, 0.28113237, 0.16851817, 0.2, 0.2, 0.17603329, 
            0.21581194, 0.2, 0.2928223, 0.16747166, 1.45242466, 1.45812127, 0.69531447, 1.1, 1.1, 1.1])
        
        # Creating a realsense pipeline
        # for cam in self.cfg.realsense.serial_numbers:
        self.pipeline, config = camera.create_realsense_pipeline(self.cfg.realsense.serial_numbers[3], self.cfg.realsense.resolution, self.cfg.realsense.fps)
        print(config)
        self.pipeline.start()
        # self.pipeline.start(config)

        # Creating mediapipe objects
        self.mediapipe_drawing = mediapipe.solutions.drawing_utils
        self.mediapipe_hands = mediapipe.solutions.hands

        # Initializing the Alegro Inverse Kinematics based controller
        self.allegro_control = AllegroIKController(self.cfg.allegro,os.path.join(os.getcwd(),'urdf_template','allegro_right.urdf'))

        # Joint state publisher object
        # self.allegro_pub = AllegroEnv()

        # Initializing a current joint state variable and creating a subscriber to get the current allegro joint angles
        self.current_joint_state = np.ones(16) * 0.2
        # rospy.Subscriber(self.cfg.allegro.joint_angle_topic, JointState, self._callback_current_joint_state, queue_size = 1)

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
        if(self.record_demo):
            self.initialize_demo_dict(initial_env)

    


    def initialize_demo_dict(self, start_state):
        #TODO double check what this means...
        self.demo_dict['terminated'] = True
         #TODO initialize this before first step
        self.demo_dict['init_state_dict']= {}       
        self.demo_dict['init_state_dict'] = {'desired_orien':self.env.env.desired_orien,
                                            'qvel':self.env.env.sim.data.qvel,
                                            'qpos':self.env.env.sim.data.qpos}

        self.demo_dict['env_infos'] = {'qvel':None, 'desired_orien':None, 'qpos':None}
        self.demo_dict['observations'] = None
        self.demo_dict['rewards'] = None    
        self.demo_dict['actions'] = None    



    def add_im(self, image):
        #TODO maybe add an option to save each individual frame? TBD
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

    def check_bounds_to_compute_angles(self, wrist_position, pinky_knuckle_position, finger_tip_positions):
        # desired_angles = np.ones(16) * 0.6
        # Check if the wrist position and pinky knuckle position are aligned
        if joint_handlers.check_hand_position(wrist_position, 
            pinky_knuckle_position, 
            self.cfg.mediapipe.contours.wrist_circle, 
            self.cfg.mediapipe.contours.pinky_knuckle_bounds):

          
            # Updating the index finger motion based on the mediapipe index finger position
            # Updating the middle fingger motion based on the mediapipe middle finger position

            # Updating the allegro ring finger angles based on the mediapipe ring finger and pinky values
            self.desired_angles = self.allegro_control.bounded_linear_fingers_motion(
                'ring', 
                self.cfg.allegro_bounds.height,         # x value for allegro hand
                finger_tip_positions['pinky'][1],       # y value for the allegro hand
                finger_tip_positions['ring'][1],        # z value for the allegro hand
                self.cfg.mediapipe_bounds.pinky_linear, 
                self.cfg.mediapipe_bounds.ring_linear, 
                self.cfg.allegro_bounds.ring[0],  
                self.cfg.allegro_bounds.ring[1], 
                self.moving_average_queues['ring'], 
                list(self.current_joint_state))


            # Checking if the thumb is inside the bound
            # if cv2.pointPolygonTest(np.array(self.cfg.mediapipe_bounds.thumb), finger_tip_positions['thumb'], False) > -1:
                # Getting the transformed thumb tip coordinate using projective transformation on the finger tip coordinate
            transformed_thumb_coordinate = perform_persperctive_transformation(
                finger_tip_positions['thumb'], 
                self.cfg.mediapipe_bounds.thumb,
                self.cfg.allegro_bounds.thumb,
                self.cfg.allegro_bounds.height)
            # import pdb
            # pdb.set_trace()
            print(transformed_thumb_coordinate)
            # Finding the desired thumb angles and updating them on the ring finger angles using the bounded inverse kinematics finction
            self.desired_angles = self.allegro_control.bounded_finger_motion(
                'thumb', 
                transformed_thumb_coordinate, 
                self.moving_average_queues['thumb'], 
                self.desired_angles)
            # else:
            #     print('thumb not inside bounds!')

            return self.desired_angles
        return 

    def hand_movement_processor(self):
        # Setting the mediapipe hand parameters
        with self.mediapipe_hands.Hands(
            max_num_hands = 1, # Limiting the number of hands detected in the image to 1
            min_detection_confidence = 0.75,
            min_tracking_confidence = 0.75) as hand:

            while True:
                # Getting the image to process
                image = camera.getting_image_data(self.pipeline)

                if image is None:
                    print('Did not receive an image. Please wait!')
                    continue

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
                    print("hand detected")
                    # Getting the hand coordinate values for the only detected hand
                    hand_landmarks = estimate.multi_hand_landmarks[0]

                    # Embedding the hand drawind in the image
                    self.mediapipe_drawing.draw_landmarks(
                            image, hand_landmarks, self.mediapipe_hands.HAND_CONNECTIONS)

                    if self.current_joint_state is not None:
                        # Getting the mediapipe wrist and fingertip positions
                        wrist_position, thumb_knuckle_position, index_knuckle_position, middle_knuckle_position, ring_knuckle_position, pinky_knuckle_position, finger_tip_positions = joint_handlers.get_joint_positions(hand_landmarks, self.cfg.realsense.resolution, self.cfg.mediapipe)
                        
                        
                        
                        # Getting the desired angles based on the current joint and tip positions
                        self.desired_angles = self.check_bounds_to_compute_angles(wrist_position, 
                                pinky_knuckle_position, 
                                finger_tip_positions)
                        

                        if self.desired_angles is not None:
                            self.desired_angles[1], self.desired_angles[2] = joint_handlers.get_approx_index(hand_landmarks)
                            self.desired_angles[5],self.desired_angles[6] = joint_handlers.get_approx_middle(hand_landmarks)
                            # Using the ROS publisher to publish the angles
                            # self.allegro_pub.pose_step(desired_angles)

                            print("Desired angles: ", self.desired_angles)                        
                            joints = np.array(self.desired_angles) 
                            self.obs_ctr += 1      
                            self.current_joint_state =  np.array(self.desired_angles) 
                            
                            
                #check if block needs to be reset
                num_resets = 0
                while(not self.env.env.is_on_palm()):
                    num_resets += 1
                    if(num_resets <= 5):
                        joints = np.ones(16) *0.2
                        for i in range(20):
                            obs,reward, done, info = self.env.env.step(joints)
                            if(self.display_window): self.env.env.render()
                        
                        for i in range(5):
                            self.env.env.reset_model()
                            if(self.display_window): self.env.render()
                        num_resets = 0
                    print('RESETTING BLOCK')
                    import pdb
                    # pdb.set_trace()
                    self.env.env.reset_model()
                # print("joints")
                print(self.current_joint_state)
                for i in range(10):
                    obs,reward, done, info = self.env.step(self.current_joint_state)
                if(self.record_demo):
                    self.unmrkd_images.append(raw_im)
                    self.add_im(image)
                if(self.record_demo and self.obs_ctr % self.obs_freq == 0):
                    self.add_demo_entry(self.env, self.current_joint_state, obs,reward, done, info)
                if(self.record_demo and self.obs_ctr % (self.obs_freq * 100) == 0):
                    print('SAVING DEMO...')
                    self.finish_recording()
                    

                # Printing the image
                if(self.display_window):
                    cv2.imshow('Teleop - Mediapipe screen', image)
                    self.env.render()
                
                    
                # self.env.env.reset_model()
                # Condition to break the loop incase of keyboard interrupt
                if cv2.waitKey(30) & 0xFF == 27:
                    
                    if(self.record_demo):
                        print('do not interrupt...saving demo...')
                        self.finish_recording()
                        print(self.demo_dir)
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use teleop to operate Mujoco sim')
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--headless', type=bool, default=False)
    args = parser.parse_args()

    teleop = TeleOpSim(args.record, args.headless, rotation_angle = 180)
    # embed()
    try:
        teleop.hand_movement_processor()
    except KeyboardInterrupt:
        print ('Interrupted')
        if(teleop.record_demo):
            teleop.finish_recording()
            print(teleop.demo_dir)
        sys.exit(0)
