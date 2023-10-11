# imports
import numpy as np


def B_dipole(coords, sph=False):
    """
    Compute Earth's dipole magnetic field at given cartesian coordinates.

    Args:
        coords (list): array of cartesian coordinates (x, y, z) in units [RE]
        sph (bool, optional):  deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        float: value of dipole field in units [T] or [kg*s^-2*A^-1]
    """
    if sph == True:
        # unpack coordinates
        rdir = coords[0]
        theta = coords[1]
        phi = coords[2]

        # convert spherical to cartesian coordinates
        x = rdir * np.sin(theta) * np.cos(phi)  # x-coordinate
        y = rdir * np.sin(theta) * np.sin(phi)  # y-coordinate
        z = rdir * np.cos(theta)  # z-coordinate

        r = np.array([x, y, z])

    else:
        r = coords

    # tilt of magnetic axis [rad]
    phi = np.radians(11.7)

    # magnetic moment of earth [A*m^2]
    mu = -7.94e22 * np.array([0.0, np.sin(phi), np.cos(phi)])

    # mu_0 [N*A−2] /4*PI
    M0 = 1.0e-7

    rmag = np.sqrt(np.dot(r, r))

    # calculating dipole field
    Bdip = M0 * (3.0 * r * np.dot(mu, r) / (rmag**5) - mu / (rmag**3))
    return Bdip
