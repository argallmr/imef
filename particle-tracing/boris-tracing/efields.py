import numpy as np

def crt_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2) # radial distance
    theta = np.arccos(z/r)          # polar angle
    phi = np.arctan2(y,x)           # azimuthal angle

    return r, theta, phi

def convection_field(kp):
    """
    Compute uniform convection electric field strength in equatorial plane in [kV] with given kp index.

    Args:
        kp (float): kp index

    Returns:
        float: convection electric field
    """
    # equatorial radius of earth [km] 
    RE = 6371
     
    # uniform convection electric field strength in equatorial plane [kV]
    E0 = 0.045/((1-(0.159*kp) + (0.0093*kp**2))**3*(RE**2.))
    
    return E0


def vs_efield(coords, gs, kp, sph = False):
    """
    Compute Volland-Stern electric field in [mV/m]

    Args:
        coords (list):          array of coordinates to compute efield in [x,y,z] or [r, theta, phi]; see condition.
        gs (floay):             shielding constant
        kp (float):             kp index
        sph (bool, optional):   Deterimes if input coordinates are in cartesian (False) or spherical (True); Defaults to False.

    Returns:
        list: volland-stern field in 3 dimensions.
    """
    
    if sph == False:
        # convert cartesian to spherical coordinates
        rgeo, theta, phi = crt_to_sph(coords[0], coords[1], coords[2])

    else:
        # unpack coordinates
        rgeo  = coords[0]
        theta = coords[1]
        phi   = coords[2]

    
    # uniform convection electric field strength in equatorial plane [kV]
    E0 = convection_field(kp)

    # VS efield [kV/km]
    EC0 = E0*gs*(rgeo**(gs-1))*(np.sin(phi))                        # radial componenet
    EC1 = 0.                                                        # polar component
    EC2 = E0*(rgeo**(gs-1))*(1/(np.sin(theta)))*(np.cos(phi))       # azimuthal component

    # set to array and convert to [mV/m]
    EC = np.array([EC0, EC1, EC2])*1e3
    
    return EC