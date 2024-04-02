""" Global definitions """

import os

import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__))
ASSET_PATH = os.path.join(PROJECT_ROOT, "assets")
MESH_PATH = os.path.join(ASSET_PATH, "meshes")

AXIS2ARRAY = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}