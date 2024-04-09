"""Implements model class for hand with 5 fingers

"""
import numpy as np
from logger import getlogger
from model.defaults import DEFAULTS

from .model import Model
from .utils import SafeFString

logger = getlogger(__name__)

# TODO: Check if we can remove the distal bracket geometry


class Model3D5FH_UR5_PG(Model):
    """Implements model for hand with 5 fingers"""

    NUM_FINGERS = 5
    NUM_JOINTS = NUM_FINGERS * 3
    _object_default_height = 0.25 # define z for 3_dof or 1_dof
    _default_joint_range = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]

    _hand_meshes = SafeFString(
        """
        <asset>

            <material class="ur5e" name="black" rgba="0.033 0.033 0.033 1"/>
            <material class="ur5e" name="jointgray" rgba="0.278 0.278 0.278 1"/>
            <material class="ur5e" name="linkgray" rgba="0.82 0.82 0.82 1"/>
            <material class="ur5e" name="urblue" rgba="0.49 0.678 0.8 1"/>

            <mesh file="ur5/base.stl"/>
            <mesh file="ur5/shoulder.stl"/>
            <mesh file="ur5/upperarm.stl"/>
            <mesh file="ur5/forearm.stl"/>
            <mesh file="ur5/wrist1.stl"/>
            <mesh file="ur5/wrist2.stl"/>
            <mesh file="ur5/wrist3.stl"/>
            <mesh file="hand/proximal.stl"></mesh>
            <mesh file="hand/middle.stl"></mesh>
            <mesh file="hand/distal_bracket.stl"></mesh>
            <mesh file="hand/distal_tip.stl"></mesh>
        </asset>
        """
    )

    _hand_defaults = SafeFString(
        """
        <!--
        Hand defaults:
        - Joints are limited
        - Link meshes
        - Distal link friction
        -->

        <default>
            <default class="hand">
                <geom type="mesh"></geom>
                <joint limited="true" damping="0.15"></joint>
                <default class="prox_link">
                    <joint axis="0 1 0" range="{prox_joint_range[0]} {prox_joint_range[1]}"></joint>
                    <geom mesh="proximal" rgba="0.3 0.3 0.3 1"></geom>
                </default>
                <default class="middle_link">
                    <joint axis="-1 0 0" range="{middle_joint_range[0]} {middle_joint_range[1]}"></joint>
                    <geom mesh="middle" rgba="0.3 0.3 0.3 1"></geom>
                </default>
                <default class="dist_link">
                    <joint axis="1 0 0" range="{dist_joint_range[0]} {dist_joint_range[1]}"></joint>
                    <geom friction="1.0 0.005 0.0001" condim="4" rgba="0.2 0.2 0.2 1"></geom>
                </default>
            </default>
            <default class="position_actuator">
                <position forcelimited="true" forcerange="-2.0 2.0" ctrllimited="true" ctrlrange="-1.57079632679 1.57079632679"></position>
                <default class="prox_actuator">
                    <position kp="2" ctrlrange="-1.8849555921538759 1.8849555921538759"></position>
                </default>
                <default class="middle_actuator">
                    <position kp="2"></position>
                </default>
                <default class="dist_actuator">
                    <position kp="1"></position>
                </default>
            </default>
            <default class="ur5e">
                <material specular="0.5" shininess="0.25"/>
                <joint axis="0 1 0" range="-6.28319 6.28319" armature="0.1"/>
                <general gaintype="fixed" biastype="affine" ctrlrange="-6.2831 6.2831" gainprm="2000" biasprm="0 -2000 -400"
                    forcerange="-150 150"/>
                <default class="size3">
                    <default class="size3_limited">
                    <joint range="-3.1415 3.1415"/>
                    <general ctrlrange="-3.1415 3.1415"/>
                    </default>
                </default>
                <default class="size1">
                    <general gainprm="50" biasprm="0 -50 -10" forcerange="-28 28"/>
                </default>
                <default class="visual">
                    <geom type="mesh" contype="0" conaffinity="0" group="2"/>
                </default>
                <default class="collision">
                    <geom type="capsule" group="3"/>
                    <default class="eef_collision">
                    <geom type="cylinder"/>
                    </default>
                </default>
                <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
            </default>
        </default>

        <!-- geom user element used for position of fingertip hemisphere (along y axis) -->
        <size nuser_geom="1"></size>

        """
    )

    _arm = """
        <body name="base" pos="0 0 0" quat="1 0 0 1" childclass="ur5e">
            <inertial mass="4.0" pos="0 0 0" diaginertia="0.00443333156 0.00443333156 0.0072"/>
            <geom mesh="base" material="jointgray" class="visual"/>
            <body name="shoulder_link" pos="0 0 0.163">
                <inertial mass="3.7" pos="0 0 0" diaginertia="0.0102675 0.0102675 0.00666"/>
                <joint name="shoulder_pan_joint" class="size3" axis="0 0 1"/>
                <geom mesh="shoulder" material="urblue" class="visual"/>
                <geom class="collision" size="0.06 0.06" pos="0 0 -0.04"/>
                <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0">
                    <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.133886 0.133886 0.0151074"/>
                    <joint name="shoulder_lift_joint" class="size3"/>
                    <geom mesh="upperarm" material="linkgray" class="visual"/>
                    <geom class="collision" pos="0 -0.04 0" quat="1 1 0 0" size="0.06 0.06"/>
                    <geom class="collision" size="0.05 0.2" pos="0 0 0.2"/>
                    <body name="forearm_link" pos="0 -0.131 0.425">
                        <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0311796 0.0311796 0.004095"/>
                        <joint name="elbow_joint" class="size3_limited"/>
                        <geom mesh="forearm" material="urblue" class="visual"/>
                        <geom class="collision" pos="0 0.08 0" quat="1 1 0 0" size="0.055 0.06"/>
                        <geom class="collision" size="0.038 0.19" pos="0 0 0.2"/>
                        <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0">
                            <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0025599 0.0025599 0.0021942"/>
                            <joint name="wrist_1_joint" class="size1"/>
                            <geom mesh="wrist1" material="black" class="visual"/>
                            <geom class="collision" pos="0 0.05 0" quat="1 1 0 0" size="0.04 0.07"/>
                            <body name="wrist_2_link" pos="0 0.127 0">
                                <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0025599 0.0025599 0.0021942"/>
                                <joint name="wrist_2_joint" axis="0 0 1" class="size1"/>
                                <geom mesh="wrist2" material="black" class="visual"/>
                                <geom class="collision" size="0.04 0.06" pos="0 0 0.04"/>
                                <geom class="collision" pos="0 0.02 0.1" quat="1 1 0 0" size="0.04 0.04"/>
                                <body name="wrist_3_link" pos="0 0 0.1">
                                    <inertial mass="0.1889" pos="0 0.0771683 0" quat="1 0 0 1"
                                        diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
                                    <joint name="wrist_3_joint" class="size1"/>
                                    <geom material="linkgray" mesh="wrist3" class="visual"/>
                                    <geom class="eef_collision" pos="0 0.08 0" quat="1 1 0 0" size="0.04 0.02"/>
                                    <site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>

                                    <!-- Hand -->
                                    <geom name="hand_plate" type="cylinder" size="0.18 0.02" euler="1.5707963267948966 0 0" pos="0 0.08 0"></geom>
                                    <body name="hand_plate" euler="-1.5707963267948966 0 0" pos="0 0.04 0">
                                        <!-- Finger 1 -->
                                        <body name="finger1_proximal" childclass="prox_link" pos="0.141421356 0.0 0.041" euler="1.5707963267948966 1.5707963267948966 0">
                                            <inertial pos="0 0 0" mass="0.09313" fullinertia="0.00026367817 0.00001529521 0.00026733699 0.00000191262 -0.00000000394 -0.00000000009"></inertial>
                                            <joint name="finger1_prox_joint" ref="{joint_ref[0]}"></joint>
                                            <geom name="finger1_proximal"></geom>

                                            <body name="finger1_middle" childclass="middle_link" pos="0 0.0655 0" euler="0 0 0">
                                                <inertial pos="0 0 0" mass="0.09930" fullinertia="0.00030191767 0.00001861456 0.00030797470 0.00000203190 0.00000000003 0.00000000011"></inertial>
                                                <joint name="finger1_middle_joint" ref="{joint_ref[1]}"></joint>
                                                <geom name="finger1_middle"></geom>

                                                <body name="finger1_distal" childclass="dist_link" pos="0 0.069 0" euler="0 3.14159265359 0">
                                                    <inertial pos="0 0 0" mass="0.09268" fullinertia="0.00053149692 0.00001518193 0.00053424309 -0.00000000003 0.00000000000 0.00000474504"></inertial>
                                                    <joint name="finger1_dist_joint" ref="{joint_ref[2]}"></joint>
                                                    <geom name="finger1_distal_bracket" mesh="distal_bracket" user="0.0285"></geom>
                                                    <geom name="finger1_distal" mesh="distal_tip" pos="0 0.0285 0" user="0.075"></geom>
                                                    <site name="finger1_tip" pos="0 0.1035 0" type="box" euler="-1.57079633 0.0 -1.57079633"></site>
                                                </body>
                                            </body>
                                        </body>

                                        <!-- Finger 2 -->
                                        <body name="finger2_proximal" childclass="prox_link" pos="0.043701602 0.134499702 0.041" euler="1.5707963267948966 2.82743339 0">
                                            <inertial pos="0 0 0" mass="0.09313" fullinertia="0.00026367817 0.00001529521 0.00026733699 0.00000191262 -0.00000000394 -0.00000000009"></inertial>
                                            <joint name="finger2_prox_joint" ref="{joint_ref[3]}"></joint>
                                            <geom name="finger2_proximal"></geom>

                                            <body name="finger2_middle" childclass="middle_link" pos="0 0.0655 0" euler="0 0 0">
                                                <inertial pos="0 0 0" mass="0.09930" fullinertia="0.00030191767 0.00001861456 0.00030797470 0.00000203190 0.00000000003 0.00000000011"></inertial>
                                                <joint name="finger2_middle_joint" ref="{joint_ref[4]}"></joint>
                                                <geom name="finger2_middle"></geom>

                                                <body name="finger2_distal" childclass="dist_link" pos="0 0.069 0" euler="0 3.14159265359 0">
                                                    <inertial pos="0 0 0" mass="0.09268" fullinertia="0.00053149692 0.00001518193 0.00053424309 -0.00000000003 0.00000000000 0.00000474504"></inertial>
                                                    <joint name="finger2_dist_joint" ref="{joint_ref[5]}"></joint>
                                                    <geom name="finger2_distal_bracket" mesh="distal_bracket" user="0.0285"></geom>
                                                    <geom name="finger2_distal" mesh="distal_tip" pos="0 0.0285 0" user="0.075"></geom>
                                                    <site name="finger2_tip" pos="0 0.1035 0" type="box" euler="-1.57079633 0.0 -1.57079633"></site>
                                                </body>
                                            </body>
                                        </body>

                                        <!-- Finger 3 -->
                                        <body name="finger3_proximal" childclass="prox_link" pos="-0.114412281  0.083125388  0.041" euler="1.5707963267948966 -2.19911486 0">
                                            <inertial pos="0 0 0" mass="0.09313" fullinertia="0.00026367817 0.00001529521 0.00026733699 0.00000191262 -0.00000000394 -0.00000000009"></inertial>
                                            <joint name="finger3_prox_joint" ref="{joint_ref[6]}"></joint>
                                            <geom name="finger3_proximal"></geom>

                                            <body name="finger3_middle" childclass="middle_link" pos="0 0.0655 0" euler="0 0 0">
                                                <inertial pos="0 0 0" mass="0.09930" fullinertia="0.00030191767 0.00001861456 0.00030797470 0.00000203190 0.00000000003 0.00000000011"></inertial>
                                                <joint name="finger3_middle_joint" ref="{joint_ref[7]}"></joint>
                                                <geom name="finger3_middle"></geom>

                                                <body name="finger3_distal" childclass="dist_link" pos="0 0.069 0" euler="0 3.14159265359 0">
                                                    <inertial pos="0 0 0" mass="0.09268" fullinertia="0.00053149692 0.00001518193 0.00053424309 -0.00000000003 0.00000000000 0.00000474504"></inertial>
                                                    <joint name="finger3_dist_joint" ref="{joint_ref[8]}"></joint>
                                                    <geom name="finger3_distal_bracket" mesh="distal_bracket" user="0.0285"></geom>
                                                    <geom name="finger3_distal" mesh="distal_tip" pos="0 0.0285 0" user="0.075"></geom>
                                                    <site name="finger3_tip" pos="0 0.1035 0" type="box" euler="-1.57079633 0.0 -1.57079633"></site>
                                                </body>
                                            </body>
                                        </body>

                                        <!-- Finger 4 -->
                                        <body name="finger4_proximal" childclass="prox_link" pos="-0.114412281 -0.083125388  0.041" euler="1.5707963267948966 -0.9424778 0">
                                            <inertial pos="0 0 0" mass="0.09313" fullinertia="0.00026367817 0.00001529521 0.00026733699 0.00000191262 -0.00000000394 -0.00000000009"></inertial>
                                            <joint name="finger4_prox_joint" ref="{joint_ref[9]}"></joint>
                                            <geom name="finger4_proximal"></geom>

                                            <body name="finger4_middle" childclass="middle_link" pos="0 0.0655 0" euler="0 0 0">
                                                <inertial pos="0 0 0" mass="0.09930" fullinertia="0.00030191767 0.00001861456 0.00030797470 0.00000203190 0.00000000003 0.00000000011"></inertial>
                                                <joint name="finger4_middle_joint" ref="{joint_ref[10]}"></joint>
                                                <geom name="finger4_middle"></geom>

                                                <body name="finger4_distal" childclass="dist_link" pos="0 0.069 0" euler="0 3.14159265359 0">
                                                    <inertial pos="0 0 0" mass="0.09268" fullinertia="0.00053149692 0.00001518193 0.00053424309 -0.00000000003 0.00000000000 0.00000474504"></inertial>
                                                    <joint name="finger4_dist_joint" ref="{joint_ref[11]}"></joint>
                                                    <geom name="finger4_distal_bracket" mesh="distal_bracket" user="0.0285"></geom>
                                                    <geom name="finger4_distal" mesh="distal_tip" pos="0 0.0285 0" user="0.075"></geom>
                                                    <site name="finger4_tip" pos="0 0.1035 0" type="box" euler="-1.57079633 0.0 -1.57079633"></site>
                                                </body>
                                            </body>
                                        </body>

                                        <!-- Finger 5 -->
                                        <body name="finger5_proximal" childclass="prox_link" pos="0.043701602 -0.134499702  0.041" euler="1.5707963267948966 0.31415927 0">
                                            <inertial pos="0 0 0" mass="0.09313" fullinertia="0.00026367817 0.00001529521 0.00026733699 0.00000191262 -0.00000000394 -0.00000000009"></inertial>
                                            <joint name="finger5_prox_joint" ref="{joint_ref[12]}"></joint>
                                            <geom name="finger5_proximal"></geom>

                                            <body name="finger5_middle" childclass="middle_link" pos="0 0.0655 0" euler="0 0 0">
                                                <inertial pos="0 0 0" mass="0.09930" fullinertia="0.00030191767 0.00001861456 0.00030797470 0.00000203190 0.00000000003 0.00000000011"></inertial>
                                                <joint name="finger5_middle_joint" ref="{joint_ref[13]}"></joint>
                                                <geom name="finger5_middle"></geom>

                                                <body name="finger5_distal" childclass="dist_link" pos="0 0.069 0" euler="0 3.14159265359 0">
                                                    <inertial pos="0 0 0" mass="0.09268" fullinertia="0.00053149692 0.00001518193 0.00053424309 -0.00000000003 0.00000000000 0.00000474504"></inertial>
                                                    <joint name="finger5_dist_joint" ref="{joint_ref[14]}"></joint>
                                                    <geom name="finger5_distal_bracket" mesh="distal_bracket" user="0.0285"></geom>
                                                    <geom name="finger5_distal" mesh="distal_tip" pos="0 0.0285 0" user="0.075"></geom>
                                                    <site name="finger5_tip" pos="0 0.1035 0" type="box" euler="-1.57079633 0.0 -1.57079633"></site>
                                                </body>
                                            </body>
                                        </body>
                                    
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body> 
            </body>
        </body>
        """

    # This will be the place to set_gains
    _actuators = SafeFString(
        """
        <actuator>
        <position name="finger1_prox_joint_actuator" joint="finger1_prox_joint" class="prox_actuator" kp="{kp_prox[0]}"></position>
        <position name="finger1_middle_joint_actuator" joint="finger1_middle_joint" class="middle_actuator" kp="{kp_middle[0]}"></position>
        <position name="finger1_dist_joint_actuator" joint="finger1_dist_joint" class="dist_actuator" kp="{kp_dist[0]}"></position>
        <position name="finger2_prox_joint_actuator" joint="finger2_prox_joint" class="prox_actuator" kp="{kp_prox[1]}"></position>
        <position name="finger2_middle_joint_actuator" joint="finger2_middle_joint" class="middle_actuator" kp="{kp_middle[1]}"></position>
        <position name="finger2_dist_joint_actuator" joint="finger2_dist_joint" class="dist_actuator" kp="{kp_dist[1]}"></position>
        <position name="finger3_prox_joint_actuator" joint="finger3_prox_joint" class="prox_actuator" kp="{kp_prox[2]}"></position>
        <position name="finger3_middle_joint_actuator" joint="finger3_middle_joint" class="middle_actuator" kp="{kp_middle[2]}"></position>
        <position name="finger3_dist_joint_actuator" joint="finger3_dist_joint" class="dist_actuator" kp="{kp_dist[2]}"></position>
        <position name="finger4_prox_joint_actuator" joint="finger4_prox_joint" class="prox_actuator" kp="{kp_prox[3]}"></position>
        <position name="finger4_middle_joint_actuator" joint="finger4_middle_joint" class="middle_actuator" kp="{kp_middle[3]}"></position>
        <position name="finger4_dist_joint_actuator" joint="finger4_dist_joint" class="dist_actuator" kp="{kp_dist[3]}"></position>
        <position name="finger5_prox_joint_actuator" joint="finger5_prox_joint" class="prox_actuator" kp="{kp_prox[4]}"></position>
        <position name="finger5_middle_joint_actuator" joint="finger5_middle_joint" class="middle_actuator" kp="{kp_middle[4]}"></position>
        <position name="finger5_dist_joint_actuator" joint="finger5_dist_joint" class="dist_actuator" kp="{kp_dist[4]}"></position>
        <general class="size3" name="shoulder_pan_joint" joint="shoulder_pan_joint"/>
        <general class="size3" name="shoulder_lift_joint" joint="shoulder_lift_joint"/>
        <general class="size3_limited" name="elbow_joint" joint="elbow_joint"/>
        <general class="size1" name="wrist_1_joint" joint="wrist_1_joint"/>
        <general class="size1" name="wrist_2_joint" joint="wrist_2_joint"/>
        <general class="size1" name="wrist_3_joint" joint="wrist_3_joint"/>
    </actuator>
        """
    )