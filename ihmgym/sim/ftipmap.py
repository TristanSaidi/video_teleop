""" Implements mapping between Cartesian and AB co-ordinates for a point
on the finger tip"""

# pylint: disable=invalid-name
import numpy as np
import termcolor
from logger import getlogger

logger = getlogger(__name__)


class MappingException(ValueError):
    """Rename exception for readability"""

    pass


class FTipMap:
    """

    Provides to and fro methods for mapping between Cartesian and
    AB cordinates on the finger tip surface

    """

    r = 18e-3
    h = 62e-3
    O = np.array([0, 0, 0])
    L = r * np.sqrt(np.pi / 2)
    gamma = 1
    SCALE_xyz2ab = np.sqrt(2 * np.pi)
    SCALE_ab2xyz = 1 / np.sqrt(2 * np.pi)

    def local2ab(self, pos: np.ndarray):
        """
        Map contact position from Cartesian co-ordinates to AB
        co-ordinates

        Args:
            pos: 3D array, contact position in cartesian co-ordinates
                in the frame
        Returns:
                2D array, contact position in ab co-ordinates

        Raises:
            MappingException:

        """
        # Force input pos to be on the hemispere or the cyclindrical
        # quadrants
        if pos[2] > 0:
            pos /= np.linalg.norm(pos)
            pos *= self.r
        else:
            pos[:2] /= np.linalg.norm(pos[:2])
            pos[:2] *= self.r

        # logger.debug("%s", str(pos))
        x, y, z = pos[0], pos[1], pos[2]
        ang = np.arctan2(y, x)
        ab = None
        if z >= 0:  # red region
            # logger.debug("region: %s", termcolor.colored("red", "red"))
            ab = self._local2ab(pos)
            a = ab[0]
            b = ab[1]
            if (np.abs(a) <= self.L) is False:
                raise MappingException("a not in red region")
            if (np.abs(b) <= self.L) is False:
                raise MappingException("b not in red region")

        elif -3 * np.pi / 4 <= ang <= -np.pi / 4:  # green region
            # logger.debug("region: %s", termcolor.colored("green", "green"))
            ab = self._local2ab(np.array([pos[0], pos[1], 0.0]))  # project to edge and map to ab
            # logger.debug("z: %s", repr(z))
            ab += np.abs(z) * np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])  # "de-project"
            a = ab[0]
            b = ab[1]
            if (b <= -self.L and a + b <= 0) is False:
                raise MappingException("output does not belong to green")
        elif -np.pi / 4 < ang <= np.pi / 4:  # blue region
            # logger.debug("region: %s", termcolor.colored("blue", "blue"))
            ab = self._local2ab(np.array([pos[0], pos[1], 0.0]))  # project to edge and map to ab
            # logger.debug("z: %s", repr(z))
            ab += np.abs(z) * np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])  # "de-project"
            a = ab[0]
            b = ab[1]
            if (a >= -self.L and a + b >= 0) is False:
                raise MappingException("output does not belong to blue")
        if ab is None:
            raise MappingException("contact does not belong to red, blue or green regions")

        return ab

    def _local2ab(self, pos):
        x, y, z = pos[0], pos[1], pos[2]
        # logger.debug("ang: %s", np.rad2deg(np.arctan2(y, x)))
        if np.abs(x) >= np.abs(y):
            # logger.debug("|x| > |y|")
            a = np.sign(x) * np.sqrt(2.0 * self.r * (self.r - z)) * (np.sqrt(np.pi) / 2.0)
            b = (
                np.sign(x)
                * np.sqrt(2.0 * self.r * (self.r - z))
                * (2.0 / np.sqrt(np.pi))
                * np.arctan(y / x)
            )
        elif np.abs(y) > np.abs(x):
            # logger.debug("|y| > |x|")
            a = (
                np.sign(y)
                * np.sqrt(2.0 * self.r * (self.r - z))
                * (2.0 / np.sqrt(np.pi))
                * np.arctan(x / y)
            )
            b = np.sign(y) * np.sqrt(2.0 * self.r * (self.r - z)) * (np.sqrt(np.pi) / 2.0)
        else:
            raise MappingException("pos invalid")
        ab = np.array([a, b])
        # logger.debug("ab: %s", str(ab))
        return ab

    def ab2local(self, ab):
        """
        Compute contact position and contact normal from a,b co-ordinates
        """
        a, b = ab[0], ab[1]
        # logger.debug("ab: %s", str(ab))
        L = self.L
        if np.abs(a) < self.L and np.abs(b) < self.L:
            # logger.debug("region: %s", termcolor.colored("red", "red"))
            d = 0
            pos = self._abd2local(a, b, d)
            normal = pos / np.linalg.norm(pos)
        elif b <= -L and a + b <= 0:
            # logger.debug("region: %s", termcolor.colored("green", "green"))
            aproj, bproj = a + b + L, -L
            d = self.gamma * np.sqrt((a - aproj) ** 2 + (b - bproj) ** 2)
            pos = self._abd2local(aproj, bproj, d)
            normal = np.array([pos[0], pos[1], 0.0]) / np.linalg.norm(
                np.array([pos[0], pos[1], 0.0])
            )
        elif a >= L and a + b > 0:
            # logger.debug("region: %s", termcolor.colored("blue", "blue"))
            aproj, bproj = L, a + b - L
            d = self.gamma * np.sqrt((a - aproj) ** 2 + (b - bproj) ** 2)
            pos = self._abd2local(aproj, bproj, d)
            normal = np.array([pos[0], pos[1], 0.0]) / np.linalg.norm(
                np.array([pos[0], pos[1], 0.0])
            )
        else:
            raise MappingException("input invalid")
        return pos, normal

    def _abd2local(self, a, b, d):
        if np.abs(a) > self.L:
            raise MappingException("a out of bounds")
        if np.abs(b) > self.L:
            raise MappingException("b out of bounds")

        if np.abs(b) <= np.abs(a):
            # logger.debug("|b| < |a|")
            rho = (2 * a / np.pi) * np.sqrt(np.pi - (a / self.r) ** 2)
            phi = b * np.pi / (4 * a)
            x, y = rho * np.cos(phi), rho * np.sin(phi)
            z = self.r - ((2 * a * a) / (np.pi * self.r)) - d
        else:
            # logger.debug("|a| <= |b|")
            rho = (2 * b / np.pi) * np.sqrt(np.pi - (b / self.r) ** 2)
            phi = a * np.pi / (4 * b)
            x, y = rho * np.sin(phi), rho * np.cos(phi)
            z = self.r - ((2 * b * b) / (np.pi * self.r)) - d

        return np.array([x, y, z])

    def uniform2ab(self, ur: float, u1: float, u2: float):
        """Map uniform random variables to random a and b co-ordinates"""
        r = self.r
        h = self.h
        L = self.L
        area_red = 2 * np.pi * r * r
        area_green = area_blue = (np.pi / 2) * r * h
        areas = np.array([area_red, area_blue, area_green])
        p = np.cumsum(areas / np.sum(areas))
        if 0 <= ur < p[0]:
            # red region
            O = np.array([-L, -L])
            v1 = np.array([2 * L, 0])
            v2 = np.array([0, 2 * L])
        elif p[0] <= ur < p[1]:
            # blue region
            O = np.array([L, -L])
            v1 = h * np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])
            v2 = np.array([0, 2 * L])
        elif p[1] <= ur < p[2]:
            # green region
            O = np.array([L, -L])
            v1 = h * np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])
            v2 = np.array([-2 * L, 0])
        else:
            raise MappingException("ur must not be greater than 1")
        return O + u1 * v1 + u2 * v2