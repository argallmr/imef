# imports
import numpy as np

def B_dipole(r):
    """
    Compute Earth's dipole magnetic field at given cartesian coordinates.

    Args:
        r (list): array of cartesian coordinates (x, y, z) in units [RE]

    Returns:
        float: value of dipole field in units [T]
    """
    
    phi = np.radians(11.7)                                  # tilt of magnetic axis [rad]
    mu = -7.94e22*np.array([.0, np.sin(phi), np.cos(phi)])  # magnetic moment of earth [A*m^2]
    M0 = 1.0e-7                                             # mu_0 [N*A−2] /4*PI 
    RE = 6371000                                            # equatorial radius of Earth [m] 
    
    r = r*RE
    rmag = np.sqrt(np.dot(r, r))
    
    # calcul;ating dipole field
    Bdip = M0*(3.*r*np.dot(mu,r)/(rmag**5)-mu/(rmag**3))
    return Bdip