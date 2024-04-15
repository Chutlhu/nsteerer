import numpy as np
from collections import namedtuple

DOASetup = namedtuple('DOASetup', 'razel, distance, azimuth, elevation, deg')

def check_geo_dim_convention(x):
    assert x.shape[0] == 3
    return x

def cart2sph(xyz, deg=True):
    assert check_geo_dim_convention(xyz)
    xy = xyz[0, :]**2 + xyz[1, :]**2
    r = np.sqrt(xy + xyz[2, :]**2)
    el = np.deg2rad(90) - np.arctan2(np.sqrt(xy), xyz[2, :])
    az = np.arctan2(xyz[1, :], xyz[0, :])
    if deg:
        el = np.rad2deg(el)
        az = np.rad2deg(az)
    razel = np.array([r, az, el])
    return check_geo_dim_convention(razel)


def sph2cart(razel, deg=True):
    check_geo_dim_convention(razel)
    r, az, el = razel[0, :], razel[1, :], razel[2, :]
    if deg:
        el = np.deg2rad(el)
        az = np.deg2rad(az)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    xyz = np.array([x, y, z])
    return check_geo_dim_convention(xyz)


def _cart2sph(x,y,z):
    # DESCRIPTION: converts cartesian to spherical coordinate
    # *** INPUTS ***
    # x  (ndarray) x-coordinate(s) [N x 1]
    # y  (ndarray) y-coordinate(s) [N x 1]
    # z  (ndarray) z-coordinate(s) [N x 1]
    # *** OUTPUTS ***
    # az  (ndarray) azimuth(s) in radians [N x 1]
    # el  (ndarray) elevation(s) in radians [N x 1]
    # r   (ndarray) range(s) in radians [N x 1]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def _sph2cart(az,el,r):
    # DESCRIPTION: converts spherical to cartesian coordinate
    # *** INPUTS ***
    # az  (ndarray) azimuth(s) in radians [N x 1]
    # el  (ndarray) elevation(s) in radians [N x 1]
    # r   (ndarray) range(s) in radians [N x 1]
    # *** OUTPUTS ***
    # x  (ndarray) x-coordinate(s) [N x 1]
    # y  (ndarray) y-coordinate(s) [N x 1]
    # z  (ndarray) z-coordinate(s) [N x 1]
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z