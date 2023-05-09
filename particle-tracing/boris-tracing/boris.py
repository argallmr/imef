import numpy as np
import math
from bfields import *
from efields import *
from plottools import *


def mag(vec):
    magnitude = np.sqrt(np.dot(vec, vec))
    return magnitude


def crt_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)  # radial distance
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    return r, theta, phi


def boris(tf, r0, v0, m, q, gs, kp, tdir="fw"):
    """
    Boris particle pusher tracing code.

    Args:
        tf (int): total time in units [s]
        dt (int): timestep incrementation
        r0 (ndarray): initial position array in units [m]
        v0 (ndarray): initial velcoity array in units [m/s]
        m (float): particle mass in units [kg]
        q (float): particle charge in units [C]
        gs (float): shielding constant
        kp (float): kp index
        tdir (bool, optional): specifies forward tracing ['fw'] or backward tracing ['bw'].
                                Defaults to 'fw.'

    Returns:
        tdat (ndarray): time data in units [s]
        tdrift (ndarray): drift time in units [s]
        vdat (ndarray): velocity data in units [m/s]
        rdat (ndarray): position data in units [m]
        emag (ndarray): magnitude of total E-field in units [mV/m]
    """
    # calculate stepsize (dt) to be no bigger than half the gyroperiod
    gyroperiod = 1 / ((abs(q) * mag(B_dipole(r0))) / m)
    dt = round(0.5 * gyroperiod, 2)
    steps = int(tf / dt)

    tdat = np.zeros(steps - 1)
    rdat = np.zeros((steps, 3))
    vdat = np.zeros((steps, 3))
    emag = np.zeros(steps)

    # set initial conditions

    rdat[0] = r0  # [m]
    vdat[0] = v0  # [m/s]

    # forward vs. backward tracing
    if tdir == "fw":
        n = 1.0
    else:
        n = -1.0

    cross = False

    for i in range(0, steps - 1):
        r = rdat[i]
        v = vdat[i]

        # compute B-field [T]
        B0 = B_dipole(r)
        # B0 = np.array([0.,0.,0.])

        # compute convection E-field [mV/m]
        EC = vs_efield(r, gs, kp)

        # compute corotation E-field [mV/m]
        ER = corotation_efield(r)

        # compute total E-field and covert to [V/m]
        # E0 = np.add(EC, ER) * 1e-3
        E0 = np.array([0.0, 0.0, 0.0])

        c0 = (dt * q * B0) / (2 * m)

        # push step 1 - update velocity with half electrostatic contribution
        v1 = v + (n * (q * E0 * dt) / (2 * m))

        # push step 2 - rotated via the magnetic field contribution
        ax = v1 + (n * np.cross(v1, c0))
        bx = (2 * c0) / (1 + (n * c0**2))
        v2 = v1 + (n * np.cross(ax, bx))

        # push step 3 - updated again with another half of the electrostatic push [m/s]
        vnew = v2 + (n * (q * E0 * dt) / (2 * m))

        # update position [m]
        rnew = r + (n * (vnew * dt))

        # append to data arrays
        tdat[i] = i * dt if n == 1.0 else tf - i * dt  # time [s]
        vdat[i + 1] = vnew  # velcoity [m/s]
        rdat[i + 1] = rnew  # position [m]
        emag[i] = mag(E0 / 1e-3)  # magnitude of total E-field [mV/m]

        # calculate driftt time when particle leaves L=10

        # find position magnitude
        RE = 6371000
        rmag = np.sqrt(np.dot(rnew, rnew))

        # check if particle(s) crossed L=10
        if rmag >= 10 * RE:
            # calculate drift time [s]
            if cross == False:
                tdrift = i * dt
                # tdrift1 = (tf - (i * dt)) / 3600

            # stop calculating drift time once cross = True
            else:
                rdat[i + 1] = np.nan

            # set cross paramter so drift time is not re-calculated
            cross = True

    return tdat, tdrift, vdat, rdat, emag


def boris_backward(tf, dt, r0, v0, m, q, gs, kp):
    """
    Boris particle pusher backward tracing code.

    Args:
        tf (int): total time in units [s]
        dt (int): timestep incrementation
        r0 (ndarray): initial position array in units [m]
        v0 (ndarray): initial velcoity array in units [m/s]
        m (float): particle mass in units [kg]
        q (float): particle charge in units [C]
        gs (float): shielding constant
        kp (float): kp index

    Returns:
        tdat (ndarray): time data in units [s]
        vdat (ndarray): velocity data in units [m/s]
        rdat (ndarray): position data in units [m]
        emag (ndarray): magnitude of total E-field in units [mV/m]
    """

    steps = int(tf / dt)

    tdat = np.zeros(steps - 1)
    rdat = np.zeros((steps, 3))
    vdat = np.zeros((steps, 3))
    emag = np.zeros(steps)

    # set initial conditions

    rdat[0] = r0  # [m]
    vdat[0] = v0  # [m/s]

    for i in range(0, steps - 1):
        r = rdat[i]
        v = vdat[i]

        # compute B-field [T]
        B0 = B_dipole(r)
        # B0 = np.array([0.,0.,0.])

        # compute convection E-field [mV/m]
        EC = vs_efield(r, gs, kp)

        # compute corotation E-field [mV/m]
        ER = corotation_efield(r)

        # compute total E-field and covert to [V/m]
        E0 = np.add(EC, ER) * 1e-3

        c0 = (dt * q * B0) / (2 * m)

        # push step 1 - update velocity with half electrostatic contribution
        v1 = v - (q * E0 * dt) / (2 * m)

        # push step 2 - rotated via the magnetic field contribution
        ax = v1 - np.cross(v1, c0)
        bx = (2 * c0) / (1 - c0**2)  ##
        v2 = v1 - np.cross(ax, bx)

        # push step 3 - updated again with another half of the electrostatic push [m/s]
        vnew = v2 - (q * E0 * dt) / (2 * m)

        # update position [m]
        rnew = r - vnew * dt

        # append to data arrays
        tdat[i] = tf - i * dt  # time [s]
        vdat[i + 1] = vnew  # velcoity [m/s]
        rdat[i + 1] = rnew  # position [m]
        emag[i] = mag(E0 / 1e-3)  # magnitude of total E-field [mV/m]

    return tdat, vdat, rdat, emag
