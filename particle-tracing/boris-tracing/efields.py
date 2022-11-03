import numpy as np


def crt_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)  # radial distance
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    return r, theta, phi


""" 
corotation fields
"""


def corotation_efield(coords, sph=False):
    """
    Compute corotation electric field in equatorial plane from given coordinates.
    Based on [...]

    Args:
        coords (ndarray): array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        sph (bool, optional): deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
     ndarray: coration E-field in 3 dimensions in units [mV/m]
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]

    # equatorial magnetic field strength at surface of Earth [T]
    BE = 3.1e-5

    # angular velocity of the Earth’s rotation [rad/s]
    omega = 7.2921e-5

    # equatorial radius of earth [m]
    RE = 6371000

    # radial component with zero check
    ER0 = (omega * BE * RE**3) / rgeo**2 if rgeo != 0 else np.nan

    # convert to [mV/m]
    ER0 = ER0 * 1000

    # return array [mV/m]
    ER = np.array([ER0, 0.0, 0.0])
    return ER


def corotation_potential(coords, sph=False):
    """
    Compute corotation potential field in equatorial plane from given coordinates.

    Args:
        coords (ndarray): array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        sph (bool, optional): deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
     ndarray: magnitude of corotation potential in units [kV]
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]

    # equatorial magnetic field strength at surface of Earth [T]
    BE = 3.1e-5

    # angular velocity of the Earth’s rotation [rad/s]
    omega = 7.2921e-5

    # equatorial radius of earth [m]
    RE = 6371000

    # calculate potential [V]
    UR = -(omega * RE**3 * BE) / rgeo if rgeo != 0 else np.nan

    # return in units [kV]
    return UR * 1e-3


"""
Volland-Stern
"""


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


def vs_potential(coords, gs, kp, sph=False):
    """
    Compute Volland-Stern potential field.
    Model based on Volland 1973 (doi:10.1029/JA078i001p00171) and Stern 1975 (doi:10.1029/JA080i004p00595).

    Args:
        coords (ndarray):          array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        gs (floay):             shielding constant
        kp (float):             kp index
        sph (bool, optional):   deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        float: potential in units [kV]
    """

    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo = coords[0]
        theta = coords[1]
        phi = coords[2]

    # equatorial radius of earth [m]
    RE = 6371000

    # uniform convection electric field strength in equatorial plane [mV/m^2]
    A0 = convection_field_A0(kp)

    # VS potential [mV]
    U = -A0 * (rgeo**gs) * np.sin(phi)

    # return potential in [kv]
    return U * 1e-6


def vs_efield(coords, gs, kp, sph=False):
    """
    Compute Volland-Stern conevction electric field.
    Model based on Volland 1973 (doi:10.1029/JA078i001p00171) and Stern 1975 (doi:10.1029/JA080i004p00595).

    Args:
        coords (ndarray):          array of coordinates to compute efield in (x,y,z) or (r, theta, phi), with units in [m]; see 'sph' condition.
        gs (floay):             shielding constant
        kp (float):             kp index
        sph (bool, optional):   deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        ndarray: volland-stern field in 3 dimensions in units [mV/m].
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

    # radial componenet [mV/m]
    EC0 = A0 * gs * (rgeo ** (gs - 1)) * (np.sin(phi))

    # polar component [mV/m]
    EC1 = 0.0

    # azimuthal component [mV/m]
    EC2 = A0 * (rgeo ** (gs - 1)) * (1 / (np.sin(theta))) * (np.cos(phi))

    # return array [mV/m]
    EC = np.array([EC0, EC1, EC2])

    return EC
