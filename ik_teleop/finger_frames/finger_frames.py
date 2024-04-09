import numpy as np

def compute_normalized_position(position, empirical_range):
    """ Computes the normalized position based on the empirical range for a single point in 3D space"""
    if type(empirical_range) == dict:
        empirical_range_x, empirical_range_y, empirical_range_z = empirical_range['x'], empirical_range['y'], empirical_range['z']
    else:
        empirical_range_x, empirical_range_y, empirical_range_z = empirical_range
    # x
    mean = (empirical_range_x[0] + empirical_range_x[1]) / 2
    width = (empirical_range_x[1] - empirical_range_x[0]) / 2
    x = (position[0] - mean) / width
    # y
    mean = (empirical_range_y[0] + empirical_range_y[1]) / 2
    width = (empirical_range_y[1] - empirical_range_y[0]) / 2
    y = (position[1] - mean) / width
    # z
    mean = (empirical_range_z[0] + empirical_range_z[1]) / 2
    width = (empirical_range_z[1] - empirical_range_z[0]) / 2
    z = (position[2] - mean) / width
    return [x, y, z]


def compute_action(normalized_position, workspace_center, workspace_width, invert_mask=[False, False, False]):
    """ Computes the action based on the normalized position and the workspace bounds"""
    assert len(invert_mask) == 3
    # invert the normalized position if necessary
    for i in range(3):
        if invert_mask[i]: normalized_position[i] *= -1
    # x
    x = np.clip(normalized_position[0], -1, 1) * workspace_width[0] + workspace_center[0]
    # y
    y = np.clip(normalized_position[1], -1, 1) * workspace_width[1] + workspace_center[1]
    # z
    z = np.clip(normalized_position[2], -1, 1) * workspace_width[2] + workspace_center[2]
    return [x, y, z]



def compute_finger_frames_palm_down(hand_state_dict):
    """ Computes the normalized frame of the fingers given the hand state dictionary
    Meant for palm DOWN orientation with RIGHT hand"""
    # computes the normalized frame the fingers 
    # (z = vector from palm to finger base)
    wrist_position = hand_state_dict['wrist']
    finger_base_positions = hand_state_dict['finger_bases']

    finger_z_axes = {
        finger: np.array(finger_base_positions[finger]) - np.array(wrist_position) for finger in finger_base_positions.keys()
    }

    # normalize
    for finger in finger_z_axes:
        finger_z_axes[finger] = finger_z_axes[finger] / np.linalg.norm(finger_z_axes[finger])

    # compute y axis by taking the cross product of the z of consecutive fingers
    finger_y_axes = {
        finger: np.cross(finger_z_axes[prev_finger], finger_z_axes[finger]) for prev_finger, finger in zip(['index', 'middle', 'ring', 'pinky', 'pinky'], ['thumb', 'index', 'middle', 'ring', 'ring'])
    }
    # normalize
    for finger in finger_y_axes:
        finger_y_axes[finger] = finger_y_axes[finger] / np.linalg.norm(finger_y_axes[finger])
    
    # compute x axis by taking the cross product of the y axis and the z axis
    finger_x_axes = {
        finger: np.cross(finger_y_axes[finger], finger_z_axes[finger]) for finger in finger_y_axes.keys()
    }
    # normalize
    for finger in finger_x_axes:
        finger_x_axes[finger] = finger_x_axes[finger] / np.linalg.norm(finger_x_axes[finger])

    # construct transforms
    finger_frames = {
        finger: np.array([finger_x_axes[finger], finger_y_axes[finger], finger_z_axes[finger]]) for finger in finger_x_axes.keys()
    }

    return finger_frames


def compute_finger_frames_palm_up(hand_state_dict):
    """ Computes the normalized frame of the fingers given the hand state dictionary
    Meant for palm UP orientation with RIGHT hand"""
    # computes the normalized frame the fingers 
    # (z = vector from palm to finger base)
    wrist_position = hand_state_dict['wrist']
    finger_base_positions = hand_state_dict['finger_bases']

    finger_z_axes = {
        finger: np.array(finger_base_positions[finger]) - np.array(wrist_position) for finger in finger_base_positions.keys()
    }
    # normalize
    for finger in finger_z_axes:
        finger_z_axes[finger] = finger_z_axes[finger] / np.linalg.norm(finger_z_axes[finger])

    # compute y axis by taking the cross product of the z of consecutive fingers
    finger_y_axes = {
        key: np.cross(finger_z_axes[prev_finger], finger_z_axes[finger]) for prev_finger, finger, key in zip(['thumb', 'index', 'middle', 'ring', 'ring'], ['index', 'middle', 'ring', 'pinky', 'pinky'], ['thumb', 'index', 'middle', 'ring', 'pinky'])
    }
    # normalize
    for finger in finger_y_axes:
        finger_y_axes[finger] = finger_y_axes[finger] / np.linalg.norm(finger_y_axes[finger])
    
    # compute x axis by taking the cross product of the y axis and the z axis
    finger_x_axes = {
        finger: np.cross(finger_y_axes[finger], finger_z_axes[finger]) for finger in finger_y_axes.keys()
    }
    # normalize
    for finger in finger_x_axes:
        finger_x_axes[finger] = finger_x_axes[finger] / np.linalg.norm(finger_x_axes[finger])

    # construct transforms
    finger_frames = {
        finger: np.array([finger_x_axes[finger], finger_y_axes[finger], finger_z_axes[finger]]) for finger in finger_x_axes.keys()
    }

    return finger_frames


def get_ftip_pos_finger_frame(finger_frames, finger_tip_positions, finger_base_positions):
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