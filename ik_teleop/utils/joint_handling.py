# Standard imports
import numpy as np

# Image based imports
import cv2

def get_joint_positions(hand_landmarks, resolution, mediapipe_structure, flip = False):
    # Getting the wrist joint position in X and Y pixels 
    if flip is True:
        wrist_position = [
            int(hand_landmarks.landmark[mediapipe_structure.wrist.offset].x * resolution[0]),
            resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.wrist.offset].y * resolution[1])
        ]

        # Getting the knuckle positions in X and Y pixels
        thumb_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['thumb'].offset].x * resolution[0]),
            resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.knuckles['thumb'].offset].y * resolution[1])
        ]

        index_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['index'].offset].x * resolution[0]),
            resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.knuckles['index'].offset].y * resolution[1])
        ]

        middle_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['middle'].offset].x * resolution[0]),
            resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.knuckles['middle'].offset].y * resolution[1])
        ]

        ring_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['ring'].offset].x * resolution[0]),
            resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.knuckles['ring'].offset].y * resolution[1])
        ]

        pinky_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['pinky'].offset].x * resolution[0]),
            resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.knuckles['pinky'].offset].y * resolution[1])
        ]

        # Creating a finger tip position dictionary
        finger_tip_positions = {}

        # Getting the finger tip positions of all the fingers in the mediapipe hand
        for tip in mediapipe_structure.tips.keys():
            finger_tip_positions[tip] = [
                int(hand_landmarks.landmark[mediapipe_structure.tips[tip].offset].x * resolution[0]),
                resolution[1] - int(hand_landmarks.landmark[mediapipe_structure.tips[tip].offset].y * resolution[1])
            ]

        return wrist_position, thumb_knuckle_position, index_knuckle_position, middle_knuckle_position, ring_knuckle_position, pinky_knuckle_position, finger_tip_positions
    
    else:
        wrist_position = [
            int(hand_landmarks.landmark[mediapipe_structure.wrist.offset].x * resolution[0]),
            int(hand_landmarks.landmark[mediapipe_structure.wrist.offset].y * resolution[1])
        ]

        # Getting the knuckle positions in X and Y pixels
        thumb_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['thumb'].offset].x * resolution[0]),
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['thumb'].offset].y * resolution[1])
        ]

        index_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['index'].offset].x * resolution[0]),
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['index'].offset].y * resolution[1])
        ]

        middle_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['middle'].offset].x * resolution[0]),
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['middle'].offset].y * resolution[1])
        ]

        ring_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['ring'].offset].x * resolution[0]),
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['ring'].offset].y * resolution[1])
        ]

        pinky_knuckle_position = [
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['pinky'].offset].x * resolution[0]),
            int(hand_landmarks.landmark[mediapipe_structure.knuckles['pinky'].offset].y * resolution[1])
        ]

        # Creating a finger tip position dictionary
        finger_tip_positions = {}

        # Getting the finger tip positions of all the fingers in the mediapipe hand
        for tip in mediapipe_structure.tips.keys():
            finger_tip_positions[tip] = [
                int(hand_landmarks.landmark[mediapipe_structure.tips[tip].offset].x * resolution[0]),
                int(hand_landmarks.landmark[mediapipe_structure.tips[tip].offset].y * resolution[1])
            ]

        return wrist_position, thumb_knuckle_position, index_knuckle_position, middle_knuckle_position, ring_knuckle_position, pinky_knuckle_position, finger_tip_positions


def compute_hand_landmarks_xyz(xy_pos_dict, depth, resolution):
    # parse xy_pos_dict
    wrist_xy = xy_pos_dict['wrist']
    thumb_xy = xy_pos_dict['thumb']
    index_xy = xy_pos_dict['index']
    middle_xy = xy_pos_dict['middle']
    ring_xy = xy_pos_dict['ring']
    pinky_xy = xy_pos_dict['pinky']
    thumb_tip_xy = xy_pos_dict['finger_tip_positions']['thumb']
    index_tip_xy = xy_pos_dict['finger_tip_positions']['index']
    middle_tip_xy = xy_pos_dict['finger_tip_positions']['middle']
    ring_tip_xy = xy_pos_dict['finger_tip_positions']['ring']
    pinky_tip_xy = xy_pos_dict['finger_tip_positions']['pinky']

    # check if the xy positions are within the resolution
    for xy in [wrist_xy, thumb_xy, index_xy, middle_xy, ring_xy, pinky_xy, thumb_tip_xy, index_tip_xy, middle_tip_xy, ring_tip_xy, pinky_tip_xy]:
        if xy[0] >= resolution[0]:
            xy[0] = resolution[0] - 1
        if xy[1] >= resolution[1]:
            xy[1] = resolution[1] - 1
    # get the z positions
    wrist_pos_z = depth[wrist_xy[1], wrist_xy[0]]
    thumb_pos_z = depth[thumb_xy[1], thumb_xy[0]]
    index_pos_z = depth[index_xy[1], index_xy[0]]
    middle_pos_z = depth[middle_xy[1], middle_xy[0]]
    ring_pos_z = depth[ring_xy[1], ring_xy[0]]
    pinky_pos_z = depth[pinky_xy[1], pinky_xy[0]]
    thumb_tip_pos_z = depth[thumb_tip_xy[1], thumb_tip_xy[0]]
    index_tip_pos_z = depth[index_tip_xy[1], index_tip_xy[0]]
    middle_tip_pos_z = depth[middle_tip_xy[1], middle_tip_xy[0]]
    ring_tip_pos_z = depth[ring_tip_xy[1], ring_tip_xy[0]]
    pinky_tip_pos_z = depth[pinky_tip_xy[1], pinky_tip_xy[0]]

    # create the xyz pos dict
    xyz_pos_dict = {
        'wrist': [wrist_xy[0], wrist_xy[1], wrist_pos_z],
        'thumb': [thumb_xy[0], thumb_xy[1], thumb_pos_z],
        'index': [index_xy[0], index_xy[1], index_pos_z],
        'middle': [middle_xy[0], middle_xy[1], middle_pos_z],
        'ring': [ring_xy[0], ring_xy[1], ring_pos_z],
        'pinky': [pinky_xy[0], pinky_xy[1], pinky_pos_z],
        'thumb_tip': [thumb_tip_xy[0], thumb_tip_xy[1], thumb_tip_pos_z],
        'index_tip': [index_tip_xy[0], index_tip_xy[1], index_tip_pos_z],
        'middle_tip': [middle_tip_xy[0], middle_tip_xy[1], middle_tip_pos_z],
        'ring_tip': [ring_tip_xy[0], ring_tip_xy[1], ring_tip_pos_z],
        'pinky_tip': [pinky_tip_xy[0], pinky_tip_xy[1], pinky_tip_pos_z]
    }
    return xyz_pos_dict

def check_hand_position(wrist_position, pinky_knuckle_position, wrist_joint_bound, pinky_knuckle_bound):
    # Checking if the pinky knuckle inside the pinky knuckle contour
    return True
    if cv2.pointPolygonTest(np.array(pinky_knuckle_bound), pinky_knuckle_position, False) > -1:
        # Checking if the wrist position is within the circular bound
             
        if wrist_position[0] <= (wrist_joint_bound.center[0] + wrist_joint_bound.radius) and wrist_position[0] >= (wrist_joint_bound.center[0] - wrist_joint_bound.radius) and wrist_position[1] > (wrist_joint_bound.center[1] - wrist_joint_bound.radius):
            return True
        else:
            print('check wrist!')
    else:
        print('check pinky!')
    return False

def get_approx_index(hand_landmarks):
    l0 = hand_landmarks.landmark[0]
    wrist_base = np.array((l0.x,l0.y,l0.z))

    l5 = hand_landmarks.landmark[5]
    index_base = np.array((l5.x,l5.y,l5.z))

    l6 = hand_landmarks.landmark[6]
    index_knuckle = np.array((l6.x,l6.y,l6.z))

    l7 = hand_landmarks.landmark[7]
    index_distal= np.array((l7.x,l7.y,l7.z))

    v0 = index_base - wrist_base
    v1 = index_knuckle - index_base
    v2 = index_distal - index_knuckle

    base_angle = np.arccos(np.dot(v0, v1 )/ (np.linalg.norm(v0) * np.linalg.norm(v1)))
    base_angle = np.clip(base_angle,0,1.6)

    index_angle = np.arccos(np.dot(v1, v2 )/ (np.linalg.norm(v1) * np.linalg.norm(v2)))
    index_angle = np.clip(index_angle,0,1.6)
    print (index_angle)
    return base_angle, index_angle



    
def get_approx_middle(hand_landmarks):
    l0 = hand_landmarks.landmark[0]
    wrist_base = np.array((l0.x,l0.y,l0.z))

    l9 = hand_landmarks.landmark[9]
    middle_base = np.array((l9.x,l9.y,l9.z))

    l10 = hand_landmarks.landmark[10]
    mid_kn = np.array((l10.x,l10.y,l10.z))

    l11 = hand_landmarks.landmark[11]
    mid_dist= np.array((l11.x,l11.y,l11.z))

    v0 = middle_base - wrist_base
    v1 = mid_kn - middle_base
    v2 = mid_dist - mid_kn

    base_mid_angle = np.arccos(np.dot(v0, v1 )/ (np.linalg.norm(v0) * np.linalg.norm(v1)))
    base_mid_angle = np.clip(base_mid_angle,0,1.5)

    mid_angle = np.arccos(np.dot(v1, v2 )/ (np.linalg.norm(v1) * np.linalg.norm(v2)))
    mid_angle = np.clip(mid_angle,0,1.5)
    print (mid_angle)
    return base_mid_angle, mid_angle