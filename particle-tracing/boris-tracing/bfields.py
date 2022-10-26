# imports
import numpy as np


def B_dipole(r):
    """
    Compute Earth's dipole magnetic field at given cartesian coordinates.

    Args:
        r (list): array of cartesian coordinates (x, y, z) in units [m]

    Returns:
        float: value of dipole field in units [T] or [kg*s^-2*A^-1]
    """

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
