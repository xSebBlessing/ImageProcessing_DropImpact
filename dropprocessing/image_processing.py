import numpy as np
import cv2

from matplotlib import pyplot as plt


def normalize_images(images: list[np.ndarray], dynamic_range: int) -> np.ndarray:
    """
    normalized the grayscale images to [0; 1]

    Args:
        images (list[np.ndarray]): image data set
        dynamic_range (int): dynamic range of camera in bit

    Returns:
        normalized images (np.ndarray): grayscale images in range [0; 1]
    """
    max_grayscale_val = pow(2, dynamic_range)
    return np.true_divide(images, max_grayscale_val)


def create_bg(images: list[np.ndarray]) -> np.ndarray:
    """
    calculate background image by finding the each brightest row over all
    images and and putting those brightest rows together

    Args:
        images (list[np.ndarray]): image data set

    Returns:
        bg (np.ndarray): background image
    """
    bg = np.zeros_like(images[0])

    for image in images:
        im_mean = np.mean(image, axis=0)
        bg_mean = np.mean(bg, axis=0)
        brighter_than = im_mean > bg_mean

        bg[:, brighter_than] = image[:, brighter_than]

    return bg


def detect_ground(bg_image: np.ndarray, visualise: bool = False) -> int:
    """
    find the ground row by finding the minimum of the mean vertical grayscale
    gradient.

    Args:
        bg_image (np.ndarray): background image
        visualise (bool): if True, plot the mean vertical grayscale gradient

    Returns:
        ground_row_index (int): index of the ground row
    """
    grady = cv2.Sobel(bg_image, -1, 0, 1, ksize=15,
                        borderType=cv2.BORDER_REPLICATE)

    grady_row_mean = np.mean(grady, axis=1)

    ground_row = np.argmin(grady_row_mean)

    if visualise:
        fig, ax = plt.subplots()
        ax.plot(np.arange(grady_row_mean.size), grady_row_mean)
        ax.set_title("Mean vertical gradient by row")
        ax.set_xlabel("Row [px]")
        ax.set_ylabel("Mean vertical gradient")
        plt.show()
    
    return ground_row


def coarse_crop(images: list[np.ndarray], top: int, left: int, bottom: int,
                right: int) -> list[np.ndarray]:
    """
    Crops the (manually) given left right top and bottom margins from the images

    Args:
        images (list[np.ndarray]): image data set
        top (int): crop margin top [px]
        left (int): crop margin left [px]
        bottom (int): crop margin bottom [px]
        right (int): crop margin right [px]

    Returns:
        cropped_images (list[np.ndarray]): coarsly cropped images
    """

    dims = images[0].shape
    width = dims[1]
    height = dims[0]

    out = []
    for image in images:
        out.append(image[top:height-bottom+1, left:width-right+1])

    return out


def cut_ground_rows(images, ground_row):
    """
    removes the ground rows from image

    Args:
        images (_type_): image data set
        ground_row (_type_): index of the ground row

    Returns:
        images_ground_removed (list[np.ndarray]): images without ground
    """
    return [image[:ground_row] for image in images]


def center_drop(images: list[np.ndarray], bg: np.ndarray,
                margin: int) -> tuple[list[np.ndarray], int, int]:
    """
    centers the droplet by detecting the leftmost and rightmost point it
    reaches and cropping until there with a certain margin left.


    Args:
        images (list[np.ndarray]): image data set
        bg (np.ndarray): background image
        margin (int): number of rows to keep outside of droplet left and right

    Returns:
        out (list[np.ndarray]): images with drop centered
        leftmost_idx (int): image index with leftmost drop border
        rightmost_idx (int): image index with rightmost drop border
    """

    # initialize leftmost on the right side and rightmost on the left side of
    # the image.
    leftmost = images[0].shape[1]
    rightmost = 0
    
    # pick a binarization threshold based on the mean background brightness
    thresh = 0.5 * np.mean(bg)

    # excluding first 30 images (mostly empty images) to prevent distortion
    for idx, image in enumerate(images[30:]):
        # for each image, subtract the background and then binarize the image
        # background subtraction is used to prevent background influence.
        image = image[:-5, :] - bg[:-5, :]
        image = image < -thresh

        # calculate horizontal grayscale gradient since we want to find the 
        # boundary of the droplet on the ground
        gradx = cv2.Sobel(np.float32(image), -1, 1, 0, ksize=3,
                         borderType=cv2.BORDER_REPLICATE)

        # rightmost minimum of gradient in a row is the right drop border
        # leftmost maximum of gradient in a row is left drop border
        minima = np.argmin(gradx, axis=1)
        maxima = np.argmax(gradx, axis=1)

        # for all rows, find the rightmost and leftmost extent of the drop
        for minimum, maximum in zip(minima, maxima):
            # zero excluded since argmax will be given as 0 if no gradient is
            # present
            if maximum < leftmost and not maximum == 0:
                leftmost = maximum
                leftmost_idx = idx+30
            if minimum > rightmost:
                rightmost = minimum
                rightmost_idx = idx+30

    left_bound =  max(leftmost-margin, 0)
    right_bound = min(rightmost+margin, images[0].shape[1])

    # cut images to determined bounds
    out = []
    for image in images:
        out.append(image[:, left_bound:right_bound])
    
    return out, leftmost_idx, rightmost_idx


def split_images(images: list[np.ndarray], center_cols_to_omit: int) \
                 -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """
    Splits the images into left and right halfs, omitting a specified number of
    columns in the center to not consider the highlight in the droplet center

    Args:
        images (list[np.ndarray]): image data set
        center_cols_to_omit (int): number of columns in the center to leave out

    Returns:
        left (list[np.ndarray]): left halfs of the input images
        right (list[np.ndarray]): right halfs of the input images
    """
    imwidth = images[0].shape[1]

    right_half_offset = imwidth//2+center_cols_to_omit

    left = [image[:, :right_half_offset-2*center_cols_to_omit]
            for image in images]
    right = [image[:, right_half_offset:] for image in images]

    return left, right, right_half_offset
