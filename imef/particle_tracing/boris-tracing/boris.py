import numpy as np
import math
from bfields import *
from efields import *
from plottools import *
from tqdm import tqdm


# compute vector of numpy array
def mag(vec):
    magnitude = np.sqrt(np.dot(vec, vec))
    return magnitude


def crt2sph(x_ijk):
    r_ijk = np.zeros_like(x_ijk)
    r_ijk[:,0] = np.sqrt(np.sum(x_ijk**2, axis=1))
    r_ijk[:,1] = np.arccos(x_ijk[:,2] / r_ijk[:,0])
    r_ijk[:,2] = np.arctan2(x_ijk[:,1], x_ijk[:,0])
    return r_ijk

# convert cartesian to spherical coordinates
def crt_to_sph(x, y, z):
    """
    (x, y, z) to (r, theta, phi)
    r:      radial distance
    theta:  polar angle (radians)
    phi:    azimuthal angle (radians)
    """
    r = np.sqrt(x**2 + y**2 + z**2)  # radial distance
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    return r, theta, phi

def sph2cart(r_ijk):
    x_ijk = np.zeros_like(r_ijk)
    x_ijk[:,0] = r_ijk[:,0] * np.sin(r_ijk[:,1]) * np.cos(r_ijk[:,2])
    x_ijk[:,1] = r_ijk[:,0] * np.sin(r_ijk[:,1]) * np.sin(r_ijk[:,2])
    x_ijk[:,2] = r_ijk[:,0] * np.cos(r_ijk[:,1])
    return x_ijk

# convert spherical to cartesian coordinates
def sph_to_crt(r, theta, phi):
    """
    (r, theta, phi) to (x,y,z)
    r:      radial distance
    theta:  polar angle (radians)
    phi:    azimuthal angle (radians)
    """
    x = r * np.sin(theta) * np.cos(phi)  # x-coordinate
    y = r * np.sin(theta) * np.sin(phi)  # y-coordinate
    z = r * np.cos(theta)  # z-coordinate

    return x, y, z


def boris(tf, r0, v0, m, q, gs, kp, tdir="fw", rmax=10, dn_save=1, dt_wci=0.01):
    """
    Boris particle pusher tracing code.

    Args:
        tf (int): total time in units [s]
        dt (int): timestep incrementation
        r0 (ndarray): initial position array in units [RE]
        v0 (ndarray): initial velcoity array in units [m/s]
        m (float): particle mass in units [kg]
        q (float): particle charge in units [C]
        gs (float): shielding constant
        kp (float): kp index
        tdir (bool, optional): specifies forward tracing ['fw'] or backward tracing ['bw'].
                                Defaults to 'fw.'
        rmax (float, optional): maximum radial distance a particle is allowed to reach

    Returns:
        tdat (ndarray): time data in units [s]
        tdrift (ndarray): drift time in units [s]
        vdat (ndarray): velocity data (spherical) in units [m/s]
        rdat (ndarray): position data (spherical) in units [m]
        emag (ndarray): magnitude of total E-field in units [mV/m]
    """
    RE = 6371000
    
    # calculate stepsize (dt) to be no bigger than half the gyroperiod
    gyroperiod = (2*np.pi) / ((abs(q) * mag(B_dipole(r0*RE))) / m)
    dt = dt_wci * gyroperiod  # round(0.5 * gyroperiod, 2)
    # print("tf,dt, r0, q, m", gyroperiod)
    steps = int(tf / dt)
    nout = steps // dn_save

    print('Run time: {0}, Time step: {1}, Steps: {2}'
          .format(tf, dt, steps))

    # old
    tdat = np.zeros((nout,))
    rdat = np.zeros((nout, 3)) * np.nan
    vdat = np.zeros((nout, 3))
    emag = np.zeros((nout,))  # * np.nan
    # print("tdat =", tdat[1])

    # tdat = np.array([np.nan] * steps - 1)
    # rdat = np.array(([np.nan] * steps), 3)
    # vdat = np.array(([np.nan] * steps), 3)
    # emag = np.array([np.nan] * steps)

    # set initial conditions
    tdat[0] = 0
    rdat[0] = r0  # [RE]
    vdat[0] = v0  # [RE/s]
    isave = 1     # We have already saved the first data point at t=0
    rnew = r0
    vnew = v0

    # forward vs. backward tracing
    if tdir == "fw":
        n = 1.0
    else:
        n = -1.0

    for i in tqdm(range(0, steps - 1)):
        # print('{0} of {1}; Saving {2} of {3}'.format(i, steps, int(i // dn_save), nout))

        # set current position and velocity (cartesian coords)
        r = rnew # rdat[i]
        v = vnew # vdat[i]

        # compute B-field [T]
        B0 = B_dipole(r*RE, sph=False)
        # test function: no B-field
        # B0 = np.array([0.0, 0.0, 0.0])

        # print("r0, v0", r, v)
        # print("B", B0)

        # compute convection E-field [mV/m]
        EC = vs_efield(r, gs, kp, sph=False)

        # compute corotation E-field [mV/m]
        ER = corotation_efield(r, sph=False)

        # compute total E-field and covert to [V/m]
        # E0 = np.add(EC, ER) * 1e-3
        E0 = np.array([0.0, 0.0, 0.0])

        # c0, ax, bx - arbitrary; to break down equation
        c0 = (dt * q * B0) / (2 * m)

        # push step 1 - update velocity with half electrostatic contribution
        v1 = v + (n * (q * E0 * dt) / (2 * m))

        # push step 2 - rotated via the magnetic field contribution
        ax = v1 + (n * np.cross(v1, c0))
        bx = (2 * c0) / (1 + (n * c0**2))
        v2 = v1 + (n * np.cross(ax, bx))

        # push step 3 - updated again with another half of the electrostatic push [m/s]
        vnew = v2 + (n * (q * E0 * dt) / (2 * m))

        # update position [RE]
        rnew = r + (n * (vnew * dt) / RE)

        # Append to data arrays
        #   - Iteration i creates data for i+1
        if ((i+1) % dn_save) == 0:
            # print('Saving {0} of {1}'.format(i // dn_save, nout))
            tdat[isave] = (i+1) * dt  # if n == 1.0 else tf - i * dt  # time [s]
            vdat[isave] = vnew  # velcoity [m/s]
            rdat[isave] = rnew  # position [RE]
            emag[isave] = mag(E0 / 1e-3)  # magnitude of total E-field [mV/m]
            isave += 1

        # calculate driftt time when particle leaves L=10

        # find positional magnitude
        rmag = np.sqrt(np.dot(rnew, rnew))
        # """
        # check if particle(s) crossed rmax
        if rmag >= rmax:
            tdrift = (i + 1) * dt
            break
        else:
            tdrift = np.nan
    # """

    # Trim the data if rmag > rmax
    if isave < nout:
        tdat = tdat[:isave]
        rdat = tdat[:isave,:]
        vdat = tdat[:isave,:]
        edat = tdat[:isave,:]

    # tdrift = 1.0

    return tdat, tdrift, vdat, rdat, emag
