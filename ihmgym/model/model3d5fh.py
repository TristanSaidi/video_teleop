"""Implements model class for hand with 5 fingers

"""
import numpy as np
from logger import getlogger
from model.defaults import DEFAULTS

from .model import Model
from .utils import SafeFString

logger = getlogger(__name__)

# TODO: Check if we can remove the distal bracket geometry


class Model3D5FH(Model):
    """Implements model for hand with 5 fingers"""

    NUM_FINGERS = 5
    NUM_JOINTS = NUM_FINGERS * 3
    _object_default_height = 0.25 # define z for 3_dof or 1_dof
    _default_joint_range = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2)]

    _hand_meshes = SafeFString(
        """
        <asset>
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
        </default>

        <!-- geom user element used for position of fingertip hemisphere (along y axis) -->
        <size nuser_geom="1"></size>

        """
    )

    _fingers = """
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
    </actuator>
        """
    )