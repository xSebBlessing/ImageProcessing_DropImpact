from collections import namedtuple
import numpy as np
import cv2

from matplotlib import pyplot as plt

import math

from scipy.optimize import curve_fit
from scipy.stats import linregress


def find_drop_impact(images: list[np.ndarray]) -> int:
    """
    Finds the frame where the drop first comes into contact with the ground 
    by counting the amount of black / dark pixels in the centers of the
    ground row.

    Args:
        images (list[np.ndarray]): image data set

    Returns:
        impact_index (int): index of image with first drop contact.
    """
    imheight = images[0].shape[0]
    imwidth = images[0].shape[1]

    center_x = imwidth // 2
    margin = 10
    ground_row = imheight - 1

    image_centers = []
    for image in images:
        image_centers.append(image[ground_row, center_x-margin:center_x+margin])

    thresh = np.mean(image_centers[0]) * 0.5

    idx = 0

    # find the index of the image where at least 15 of 20 px of the ground row
    # are dark / black, i.e. part of the droplet.
    while np.sum(image_centers[idx] < thresh) < 15:
        idx += 1
    
    return idx


def calculate_falling_velocity(images: list[np.ndarray], dt: float,
                                     px_p_m: float, impact_idx: int,
                                     visualise: bool = False):
    """
    Analyze the frames before impact to find falling velocity.

    Args:
        images (list[np.ndarray]): image data set
        dt (float): time between frames [s]
        px_p_m (float): physical spatial resolution [px/m]
        impact_idx (int): first drop contact frame
        visualise (bool): show gradient images

    Returns:
        falling velocity (float): velocity before impact.
    """

    # consider images where drop is in frame and has not yet impacted
    images_to_process = images[impact_idx-4:impact_idx]

    grads = []
    lowest = []
    for image in images_to_process:
        # consider lower part of image to omit the highlight in the drop center
        image = image[-100:-7, :]
        gradx = cv2.Sobel(np.float32(image), -1, 1, 0, ksize=5,
                          borderType=cv2.BORDER_REPLICATE)
        grady = cv2.Sobel(np.float32(image), -1, 0, 1, ksize=5,
                          borderType=cv2.BORDER_REPLICATE)

        grad = np.sqrt(np.square(gradx) + np.square(grady))

        # binarize gradient relative to gradient maximum
        grad = grad > 0.5 * np.max(grad)

        grads.append(grad)

        # find the lowet point of the drop
        maxima = np.argwhere(grad > 0)
        lowest.append(np.amax(maxima[:,0]))

    dx_px = np.diff(lowest)
    dx_m = dx_px / px_p_m
    v = dx_m / dt

    if visualise:
        fig, axs = plt.subplots(len(grads), 1)
        for ax, grad in zip(axs, grads):
            ax.imshow(grad)
        
        axs[0].set_title("Vertical gradient of images before impact")
    
    return np.mean(v)


def calculate_drop_diameter(images, px_p_m: float, impact_idx: int):
    """
    Analyze the frames before impact to find falling velocity.

    Args:
        images (list[np.ndarray]): image data set
        px_p_m (float): physical spatial resolution [px/m]
        impact_idx (int): first drop contact frame

    Returns:
        diameter (float): falling droplet diameter
    """
    rightmost = []
    leftmost = []

    # consider images where drop is in frame and has not yet impacted
    images_to_process = images[impact_idx-4:impact_idx]

    for image in images_to_process:
        gradx = cv2.Sobel(np.float32(image), -1, 1, 0, ksize=5,
                          borderType=cv2.BORDER_REPLICATE)

        maxima = np.argwhere(gradx > 0.9 * np.max(gradx))
        minima = np.argwhere(gradx < 0.9 * np.min(gradx))
        rightmost.append(np.amax(maxima[:,1]))
        leftmost.append(np.amin(minima[:,1]))
    
    d = 0
    for r, l in zip(rightmost, leftmost):
        d += r - l
    
    return d / len(rightmost) / px_p_m


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2 / (2.*sigma**2))


def calculate_boundary_points(images: list[np.ndarray], impact_idx: int,
                              rows_to_evaluate: int, boundary_offset: int) \
                              -> list[np.ndarray]:
    """
    Detect the drop boundary by analysing the gradient magnitude of a set of 
    half-droplet images. The initial guess (max. gradient) is refined by fitting
    a Gaussian to the gradient values of the initial guess position and two left
    and right of it, respectively. For right half images, the x offset from
    splitting has to be taken account for correct contact line positions global-
    ly.
    Regarding amount of rows to evaluate: More rows used allows for more stable
    contact angle calculation. However, the linearization of the contact line
    becomes worse with every additional evaluated row.

    Args:
        images (list[np.ndarray]): image data set
        impact_idx (int): index of frame with drop impact
        rows_to_evaluate (int): number of lower rows to evaluate.

    Raises:
        RuntimeError: Fit did not converge, most likely due to noisy data
        ValueError: most likely left or right bound is out of range in the array

    Returns:
        boundaries (list[np.ndarray]): list of arrays with y-x-pairs of boundary
                                        line position
    """
    # analyze the bottom rows for boundary
    images_to_process = [image[-rows_to_evaluate:, :]
                         for image in images[impact_idx:]]

    # settings for Gauss fit
    p0 = [1., 0., 1.]  # initial guess coeffs
    x = np.arange(-2, 3)  # space to evaluate function
    x_lin = np.linspace(-2, 2, 2000)  # space to find max position in

    refined_boundaries = []
    for image_idx, image in enumerate(images_to_process):
        # for each image, calculate the magnitude of the grayscale gadient
        gradx = cv2.Sobel(np.float32(image), -1, 1, 0, ksize=-1,
                          borderType=cv2.BORDER_REPLICATE)
        grady = cv2.Sobel(np.float32(image), -1, 0, 1, ksize=-1,
                          borderType=cv2.BORDER_REPLICATE)
        grad = np.sqrt(np.square(gradx) + np.square(grady))

        # maximum gradient value for each row as initial guzess for boundary pos
        initial_boundary = np.argmax(grad, axis=1)

        # initialise array for refined boundary
        refined_boundary = np.empty((rows_to_evaluate, 2))

        # for each considered row, find the subpixel accurate boundary position
        # by fitting a Gaussian function to the distribution of greyscale 
        # gradient in horizontal direction
        for i, initial_boundary_point in enumerate(initial_boundary):
            y = i + images[0].shape[0] - rows_to_evaluate
            left_bound = initial_boundary_point-2
            right_bound = initial_boundary_point+3
            try:
                coeff, _ = \
                    curve_fit(gauss, x, grad[i, left_bound:right_bound], p0=p0)
            except RuntimeError:
                raise RuntimeError
            except ValueError:
                raise ValueError
            
            try:
                refined_boundary[i, :] = \
                    np.array([y, x_lin[np.argmax(gauss(x_lin, *coeff))]
                              + initial_boundary[i] + boundary_offset])
            except UnboundLocalError:
                # if the least squares fit for the Gaussian failed, use the 
                # initial guess instead
                refined_boundary[i, :] = \
                    np.array([y, initial_boundary[i] + boundary_offset])
        
        refined_boundaries.append(refined_boundary)

    return refined_boundaries
    
def calculate_contact_angles(boundaries: list[np.ndarray], side: str) \
                             -> tuple[np.ndarray, list[namedtuple]]:
    """
    Calculate contact angles from detected boundary by fitting a linear function
    through it. The linear approximation of the contact line becomes worse with
    additional evaluated rows, however, the process is less prone to distortion.

    Args:
        boundaries (list[np.ndarray]): List of arrays with y-x-boundary positions.
        side (str): 'l' or 'r', important for the correct contact angle value

    Returns:
        theta (np.ndarray): array of all contact angles
        lin_fits (list[namedtuple]): list of the LinregressResult object for visulisation.
    """
    lin_fits = []
    theta = []

    if not (side == "r" or side == "l"):
        print("ERROR: Side must be 'l' or 'r'.")

    for boundary in boundaries:
        # fit a linear function to the detected boundary
        lin_fit = linregress(boundary)

        # the slope of the approximated line is directly related to the contact
        # angle. Due to its definition (=> angle inside liquid phase), sides
        # need to be differentiated.
        if side == "l":
            angle = math.atan(lin_fit.slope) / math.pi * 180 + 90
        elif side == "r":
            angle = 90 - math.atan(lin_fit.slope) / math.pi * 180
        
        lin_fits.append(lin_fit)
        theta.append(angle)
    
    return np.array(theta), lin_fits


def calculate_spreading_velocity(x: np.ndarray, dt: float,
                                 px_p_m: float) -> np.ndarray:
    """
    Calculates the physical spreading velocity as the difference of contact
    line position between two frames over the time between frames

    Args:
        x (np.ndarray): contact line position [px]
        dt (float): time between frames [s]
        px_p_m (_type_): physical spatial resolution [px/m]

    Returns:
        velocity (np.ndarray): spreading velocity of the drop side
    """
    return np.diff(x) / px_p_m / dt

