import numpy as np


def crt_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)  # radial distance
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    return r, theta, phi


def convection_field_A0(kp):
    """
    Compute uniform convection electric field strength in equatorial plane in [kV/m^2] with given kp index.
    Based on Maynard and Chen 1975 (doi:10.1029/JA080i007p01009).

    Args:
        kp (float): kp index

    Returns:
        float: A0; convection electric field strength in [mV/m^2]
    """
    # equatorial radius of earth [m]
    RE = 6371000

    # uniform convection electric field strength in equatorial plane [kV/m^2]
    A0 = 0.045 / ((1 - (0.159 * kp) + (0.0093 * kp**2)) ** 3 * (RE**2.0))

    # convert to [mV/m^2]
    A0 = A0 * 1e6

    return A0


def vs_efield(coords, gs, kp, sph=False):
    """
    Compute Volland-Stern electric field with both convection and corotation electric field.
    Model based on Volland 1973 (doi:10.1029/JA078i001p00171) and Stern 1975 (doi:10.1029/JA080i004p00595).

    Args:
        coords (list):          array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        gs (floay):             shielding constant
        kp (float):             kp index
        sph (bool, optional):   deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        list: volland-stern field in 3 dimensions in units [mV/m].
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]
        theta = coords[1]
        phi = coords[2]

    # uniform convection electric field strength in equatorial plane [mV/m^2]
    A0 = convection_field_A0(kp)

    """VS convection efield EC [kV/m]"""
    # radial componenet [V/m]
    EC0 = A0 * gs * (rgeo ** (gs - 1)) * (np.sin(phi))

    # polar component [V/m]
    EC1 = 0.0

    # azimuthal component [V/m]
    EC2 = A0 * (rgeo ** (gs - 1)) * (1 / (np.sin(theta))) * (np.cos(phi))

    # set to array
    EC = np.array([EC0, EC1, EC2])

    """VS corotation efield ER [kV/m]"""
    # equatorial magnetic field strength at surface of Earth [T]
    BE = 4.5e-5

    # angular velocity of the Earth’s rotation [rad/s]
    omega_E = 7.2921e-5

    # equatorial radius of earth [m]
    RE = 6371000

    # radial component [V/m]
    ER0 = (omega_E * BE * RE**3) / rgeo**2

    # convert to [mV/m]
    ER0 = ER0 * 1000

    # set to array
    ER = np.array([ER0, 0.0, 0.0])

    # total electric field
    Etot = EC + ER

    return Etot
