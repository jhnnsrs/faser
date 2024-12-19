import numpy as np
from scipy.interpolate import griddata


def polar_to_cartesian(polar_image):
    """
    Convert an image in polar coordinates to Cartesian coordinates.
    :param polar_image: A 2D numpy array where the first dimension is the radius and the second is the angle.
    :return: A 2D numpy array in Cartesian coordinates.
    """
    max_radius, num_angles = polar_image.shape

    # Generate r and theta arrays
    r = np.linspace(0, max_radius, max_radius)
    theta = np.linspace(0, 2 * np.pi, num_angles)
    R, Theta = np.meshgrid(r, theta)

    # Convert polar (R, Theta) to Cartesian (X, Y)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)

    # Flatten X, Y, and polar_image for griddata interpolation
    x = X.ravel()
    y = Y.ravel()
    z = polar_image.ravel()

    # Define a grid of Cartesian coordinates
    xi, yi = np.linspace(x.min(), x.max(), polar_image.shape[1]), np.linspace(
        y.min(), y.max(), polar_image.shape[0]
    )
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate using griddata
    cartesian_image = griddata((x, y), z, (xi, yi))

    return np.angle(cartesian_image)


def polar_phase_mask(num_radii, num_angles):
    r = np.linspace(0, 1, num_radii)
    theta = np.linspace(0, 2 * np.pi, num_angles)
    R, Theta = np.meshgrid(r, theta)
    phase = (2 * np.pi * R) % (2 * np.pi)
    return phase
