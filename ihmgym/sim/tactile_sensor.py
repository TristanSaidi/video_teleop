import mujoco_py as mjpy
import numpy as np
from gym.utils import seeding
from logger import getlogger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from .ftipmap import FTipMap, MappingException

logger = getlogger(__name__)


class Profile:
    x = []
    y = []

    def __init__(self):
        self._lookup = interp1d(self.x, self.y)

    def lookup(self, force):
        return self._lookup(force)


# Noise profiles inferred from Pedro's thesis (Appendix B, page 144)
class PositionNoise(Profile):
    # Force (N)
    x = np.array([0, 0.5, 1, 2, 1000])
    # Noise (m)
    y = np.array([5, 2, 1, 1, 1]) * 1e-3


class ForceNoise(Profile):
    # Force (N)
    x = np.array([0, 0.5, 1, 2, 1000])
    # Relative Error
    y = np.array([15, 12.5, 10, 5, 5]) / 100


class TactileSensor:

    """
    Implements methods to interact with the MuJoCo simulation for
    reporting finger tip contacts with the object in finger tip frame

    """

    _ftipmap = FTipMap()

    # Rotation matrix to map from fingertip frame to frame used by
    # mapping above
    r45 = R.from_rotvec([0, 0, np.pi / 4]).as_matrix()

    def __init__(
        self,
        model,
        data,
        ftips,
        noise_scale=1,
        force_threshold=0,
        **kwargs,
    ) -> None:

        """
        model: Model object returned by mujoco_py.load_model_from_xml()
        data (PyMjData): sim.data
        ftips (list): Names of the finger tip links as a list ex.
            ["finger1_distal", "finger2_distal", ...]
        position_noise (float): stddev (in m) of the gaussian noise
            in contact position
        force_noise (float): stddev (in N) of the gaussian noise
            added to force (magnitude)
        force_noise_no_contact (float): stddev (in N) of the gaussian
            noise for spurious contact
        force_threshold (float): Threshold force (in N) below which the
            contact is not reported
        """

        self._model = model
        self._data = data
        self._ftips = ftips

        self._contacts = []

        # Noise parameters
        self._position_noise = PositionNoise()
        self._force_noise = ForceNoise()
        self._noise_scale = noise_scale
        self._force_threshold = force_threshold

        # Random number generators
        self._np_random = {}
        for name in ["contact_position", "contact_force"]:
            self._np_random[name] = np.random.default_rng()

        logger.info("Tactile sensor intialized on %s ftips", self._ftips)

    def seed(self, seed):
        seed = seeding.hash_seed(seed)
        seeds = []
        for key in self._np_random.keys():
            seeds.append(seed)
            self._np_random[key], seed = seeding.np_random(seed)
        return seeds

    def sense_contacts(self, add_noise=True):
        """
        In this method we read MuJoCo's structs to sense contacts and
        optionally add noise.

        Note: When computing the reward or computing the observation for
        value network in assymetric actor critic mode. This method
        should be called with add_noise=False

        Args:
            add_noise(boolean): As the name suggests we add noise to
            contacts detected by MuJoCo and also add spurious contacts

        Returns:
            A list contacts where entry is a dictionary consisting of
            the contact attributes in ftip frame.
        """
        contacts = []
        for ftip in self._ftips:

            # Fetch finger tip frame
            tip_pos = self._data.get_site_xpos(ftip.replace("distal", "tip"))
            tip_frame = self._data.get_site_xmat(ftip.replace("distal", "tip"))

            # Find object contact with the ftip
            contact = None
            for i in range(self._data.ncon):
                geoms = [
                    self._model.geom_id2name(geom_id)
                    for geom_id in [
                        self._data.contact[i].geom1,
                        self._data.contact[i].geom2,
                    ]
                ]
                if "object" in geoms and ftip in geoms:
                    contact = self._data.contact[i]
                    break

            # Finger tip makes contact with the object:
            ab = None
            force_magnitude = None
            frame = None
            wrench = None
            if contact:
                # Retrieve wrench as determined by MuJoCo
                frame = np.reshape(contact.frame, (3, 3)).T
                wrench = np.zeros(6, dtype=np.float64)
                # pylint: disable=E1101
                mjpy.functions.mj_contactForce(self._model, self._data, i, wrench)
                pos_local = np.dot(tip_frame.T, contact.pos - tip_pos)
                pos_mapframe = np.dot(self.r45.T, pos_local)
                # Map contact position in Cartesian co-ordinates to AB
                # co-ordinates
                # When that fails we assume no contact and move on to
                # the next finger tip. Ideally, when there is no contact
                # we need add noise (like in the else case) but we are
                # cutting corners since this is a rare occurence.
                try:
                    ab = self._ftipmap.local2ab(pos_mapframe)
                except MappingException:
                    continue
                force_magnitude = wrench[0]
                # Now that we have successfully mapped position to ab
                # co-ordinates, we will
                # add noise to both position and force.
                if add_noise:
                    position_noise = self._position_noise.lookup(force_magnitude)
                    ab = self._add_noise_to_ab(
                        ab,
                        self._noise_scale * position_noise * self._ftipmap.SCALE_xyz2ab,
                    )
                    force_noise = self._force_noise.lookup(force_magnitude)
                    force_magnitude = self._add_noise_to_force(
                        force=force_magnitude, stddev=self._noise_scale * force_noise
                    )

            # Finger tip does not make contact: We will randomly sample
            # a co-ordinate on the AB plane to compute a random contact
            # position and then also sample a force based on the
            # force threshold.
            else:
                if add_noise:
                    u = list(self._np_random["contact_position"].uniform(size=(3,)))
                    ab = self._ftipmap.uniform2ab(*u)
                    force_magnitude = self._add_noise_to_force(
                        force=0,
                        stddev=self._noise_scale * self._force_noise.lookup(0),
                    )

            # Ignore contact if the contact force is below threshold
            if (
                ab is not None
                and force_magnitude is not None
                and force_magnitude > self._force_threshold
            ):
                # Map back to Cartesian co-ordinates
                localpos, localnormal = self._ftipmap.ab2local(ab)
                localpos = np.dot(self.r45, localpos)
                localnormal = np.dot(self.r45, localnormal)

                # Append to contact
                entry = {}
                entry["link"] = ftip
                entry["position"] = localpos
                entry["normal"] = localnormal
                entry["force_magnitude"] = force_magnitude
                entry["force"] = force_magnitude * localnormal
                entry["wrench"] = np.zeros(6, dtype=np.float64)
                if frame is not None:
                    entry["frame"] = -frame
                if wrench is not None:
                    entry["wrench"] = wrench
                contacts.append(entry)

        return contacts

    def _add_noise_to_ab(self, ab, stddev):
        """Add noise to ab"""
        np_random = self._np_random["contact_position"]
        success = False
        while not success:
            r = np.abs(np_random.normal(loc=0, scale=stddev))
            ang = np_random.uniform(low=0, high=2 * np.pi)
            ab_with_noise = ab + np.array([r * np.cos(ang), r * np.sin(ang)])
            try:
                self._ftipmap.ab2local(ab_with_noise)
                success = True
            except MappingException:
                pass
        return ab_with_noise

    def _add_noise_to_force(self, force, stddev):
        return force * np.abs(
            1 + self._np_random["contact_force"].normal(scale=stddev, loc=0)
        )