""" Implements Model abstract base class """
import numpy as np
from logger import getlogger
import abc
import os
from dataclasses import dataclass

import defs

from .defaults import DEFAULTS
from .utils import SafeFString

logger = getlogger(__name__)

@dataclass
class Model(abc.ABC):
    """Abstract Base Class for model"""

    @classmethod
    def toxml(cls, **params):
        """
        Returns model XML string
        Args:
            object (str):           Specify the name of the object or the path to object mesh on
                                    disk.

                                    To specify one of the objects in the provided set specifying
                                    just the name of the object is sufficient.

                                    Specify the full path to the object mesh to load a custom mesh.

            object_size (float):    Size of the objects in meters

            object_mass (float):    Mass of the object in kg

            object_dof_type (str):  "1_dof", "3_dof, or "6_dof", to specify the degree of freedom
                                    of the object. "6_dof" will be selected if not specified.

            object_height (float):  Altitude of object, for 3_dof and 1_dof settings. Default value
                                    is 0 for 2D4FH, 0.25 for 3D5FH.

        String specifying the object or path to object mesh (.stl only). When the user choses an
        object without providing the object size, a hand tuned version of the mesh is loaded. Else,
        an unscaled mesh where the farthest vertex is at a radius of 0.5 is used with provided
        object size as scale.

        # TODO: Configurable friction (impratio)

        """
        # Generate header xml
        if "gravity" in params.keys():
            gravity = params["gravity"]
        else:
            gravity = DEFAULTS["gravity"]

        if "impratio" in params.keys():
            impratio = params["impratio"]
        else:
            impratio = DEFAULTS["impratio"]

        header_format_map = {
            "gravity": gravity,
            "meshdir": defs.MESH_PATH,
            "impratio": impratio,
        }
        header = cls._header.format_map(header_format_map)

        # Generate object xml
        if "object" not in params.keys():
            # Use empty string object if not specified
            object_model = ""
        else:
            object_format_map = cls._object_format_map(**params)
            object_model = cls._object_model.format_map(object_format_map)

        

        # Joint reference
        if "joint_ref" in params.keys():
            assert len(params["joint_ref"]) == cls.NUM_JOINTS, "joint_ref is not of correct size"
            joint_ref = np.array([ref for ref in params["joint_ref"]])
        else:
            joint_ref = np.zeros(cls.NUM_JOINTS)

        # Joint range
        joint_range_map = {}
        for key in ["prox_joint_range", "middle_joint_range", "dist_joint_range"]:
            if key in params.keys():
                joint_range_map[key] = params[key]
            else:
                joint_range_map[key] = DEFAULTS[key]

        # Fetch controller gains, set it to default value if not provided
        actuator_format_map = {}
        for key in ["kp_prox", "kp_middle", "kp_dist"]:
            if key in params.keys():
                assert isinstance(params[key], np.ndarray), "must be a np.array"
                assert params[key].ndim == 1, "must be a 1-D numpy array "
                actuator_format_map[key] = params[key]
            else:
                actuator_format_map[key] = DEFAULTS[key] * np.ones((cls.NUM_FINGERS,))

        hand_format_map = dict(
            hand_meshes=cls._hand_meshes,
            hand_defaults=cls._hand_defaults.format_map(joint_range_map),
            arm=cls._arm.format_map({"joint_ref": joint_ref}),
            actuators=cls._actuators.format_map(actuator_format_map),
        )
        hand_model = cls._hand_model.format_map(hand_format_map)

        # XML
        root_format_map = dict(
            header=header,
            hand=hand_model,
            object=object_model,
            table=cls._table,
        )
        xml = cls._root.format_map(root_format_map)

        logger.debug("XML generated")

        return xml

    _header = SafeFString(
        """
        <compiler angle='radian' coordinate='local' meshdir="{meshdir}">
        </compiler>
        <option cone='elliptic' impratio="{impratio}" tolerance='0' gravity="{gravity[0]} {gravity[1]} {gravity[2]}">
        </option>
        <visual>
            <scale forcewidth='0.01' contactwidth='0.1' contactheight='0.01'>
            </scale>
        </visual>
        <asset>
            <texture type='skybox' builtin='gradient' rgb1='1 1 1' rgb2='1 1 1' width='32' height='32'>
            </texture>
        </asset>
        """
    )

    _object_model = SafeFString(
        """
        <asset>
            <material name="object_material" specular="1" shininess="0.01" reflectance="0"></material>
            <mesh name="object_mesh" file="{object_mesh}" scale="{object_size[0]} {object_size[1]} {object_size[2]}"></mesh>
        </asset>

        <worldbody>
            <body name="object" pos="0 0 {z}" euler="0 0 0">
                {dof}
                <geom type="mesh" name="object" mesh="object_mesh" mass="{object_mass}" material="object_material">
                </geom>
                <site name="object"></site>
            </body>
        </worldbody>

        <sensor>
            <framepos objtype="body" objname="object" name="object_pos"></framepos>
            <framequat objtype="body" objname="object" name="object_quat"></framequat>
            <frameangvel objtype="body" objname="object" name="object_angvel"></frameangvel>
        </sensor>
        """
    )


    _table = """
        <worldbody>
            <body name="table" pos="0.5 0.0 0.0">
                <geom pos="0 0 0" rgba="0.1 1.0 0.1 1" size="0.4 0.75 0.1" type="box" />
            </body>
        </worldbody>
        """

    _dof_type = {
        "1_dof":
            """<joint name="rot_z" type="hinge" axis="0 0 1" frictionloss="0.01"></joint>""",
        "3_dof":
            """
            <joint name="trans_x" type="slide" axis="1 0 0" frictionloss="0.01"></joint>
            <joint name="trans_y" type="slide" axis="0 1 0" frictionloss="0.01"></joint>
            <joint name="rot_z" type="hinge" axis="0 0 1" frictionloss="0.01"></joint>
            """,
        "6_dof":
            """<joint name="object" type="free"></joint>""",
        }

    _root = SafeFString(
        """<?xml version='1.0'?>
        <mujoco model='ihm'>
        {header}

        {hand}

        {object}

        {table}
        </mujoco>
        """
    )

    _hand_model = SafeFString(
        """
        {hand_meshes}
        {hand_defaults}
        <worldbody>
        {table}
        {arm}
        </worldbody>
        {actuators}
        """
    )

    @classmethod
    def _object_format_map(cls, **params):
        """
        We evaluate the configurable parameters of the object based on params
        for different cases and return them as a dictionary.

        The configurable parameters are below:
        - object mesh
        - object size
        - object mass

        """
        # Set object mesh (absolute path)
        object_format_map = {}
        _object = params["object"]
        if os.path.isabs(_object):
            if os.path.isfile(_object.join(".stl")):
                object_mesh = _object.join(".stl")
            else:
                raise FileNotFoundError(f"""{_object.join(".stl")} not found""")
        else:
            # When the user choses an object without providing the object size, a hand tuned version
            # of the mesh is loaded. Else, an unscaled mesh where the farthest mesh is at a radius
            # of 0.5 is used with provided object size as scale.
            if "object_size" in params.keys():
                object_mesh = os.path.join(defs.MESH_PATH, "object", _object + "_unscaled.stl")
            else:
                object_mesh = os.path.join(defs.MESH_PATH, "object", _object + ".stl")
        object_format_map["object_mesh"] = object_mesh

        # Set object size
        if "object_size" in params.keys():
            object_size = params["object_size"]
            if isinstance(object_size, list):
                assert (
                    len(object_size) == 3
                ), "object_size must be a float OR a list of length size 3"
                object_format_map["object_size"] = [params["object_size"][i] for i in range(3)]
            else:
                object_format_map["object_size"] = [object_size for i in range(3)]
        else:
            object_format_map["object_size"] = [1 for i in range(3)]

        # Set object mass
        if "object_mass" in params.keys():
            object_format_map["object_mass"] = params["object_mass"]
        else:
            object_format_map["object_mass"] = DEFAULTS["object_mass"]

        # Set object dof
        if "object_dof_type" in params.keys():
            object_format_map["dof"] = cls._dof_type[params["object_dof_type"]]
        else:
            object_format_map["dof"] = cls._dof_type["6_dof"]

        # Set object z
        if "object_height" in params.keys():
            object_format_map["z"] = params["object_height"]
        else:
            object_format_map["z"] = cls._object_default_height
        return object_format_map