import os
import json

import urllib.request
import zipfile

import numpy as np
import cv2
import re

def load_processing_settings() -> tuple[str, str, str, str]:
    """loads settings for batch processing. Not included in this repo.

    Returns:
        root (str): give the data directory
        set_pattern (str): give a pattern of set directories to eval
        output_file (str): specify the name of the output file
        metadata_file (str): specify the name of the metadata file
    """
    processing_settings = {}
    with open(os.path.join("settings", "processing_settings.json"), "r") as f:
        processing_settings = json.load(f)
    
    # specify the data directory as a list of directories and concat it to path
    root_list = processing_settings["root"]
    root_list.insert(1, os.sep)
    root = os.path.join(*root_list)

    set_pattern = processing_settings["set_pattern"]
    output_file = processing_settings["output_file"]
    metadata_file = processing_settings["metadata_file"]

    return root, set_pattern, output_file, metadata_file


def load_data_set_settings(data_set_name: str) -> dict:
    """loads data set specific settings by name
       the processing routine outlined here sometimes needs some thresholds
       tweaked due to differences in recording conditions. these are stored
       in a json file and loaded using this routine.

    Args:
        data_set_name (str): identifier for the settings this data set needs

    Returns:
        dict: settings for the data set
    """
    with open(os.path.join("settings", "data_set_settings.json"), "r") as f:
        return json.load(f)[data_set_name]


def load_data_set(path: str,
                  n_images_to_process: int = -1) \
                  -> tuple[float, list[float], list[np.ndarray], dict]:
    """Load data set specified by absolute or relative path to the storage
       directory. The files must be placed inside a subdirectory named "raw"

    Args:
        path (str): relative or absolute path to data set directory
        n_images_to_process (int): first n images will be evaluated.
                                   default of -1 evaluates all images.

    Raises:
        FileNotFoundError: The metadata of the data set, specified in the file
                           "parameters.json" was not found.

    Returns:
        dt (flat): time between frames
        t (list[float]): individual image timings
        images (list[np.ndarray]): list of all images in the data set
        metadata (dict): image metadata in a dictionary
    """
    # read image meta data
    metadata = {}
    try:
        metadata_path = os.path.join(path, "parameters.json")
        with open(metadata_path, "r") as file:
            metadata = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError

    # find all tif images to read in the given path
    image_path = os.path.join(path, "raw")
    image_names = [file for file in os.listdir(image_path)
                   if file.endswith(".tif")]
    
    if not n_images_to_process == -1:
        image_names = image_names[0:n_images_to_process]

    images = []
    for image_name in image_names:
        image_file_path = os.path.join(image_path, image_name)
        images.append(cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED))
    
    dt = 1 / metadata["fps"]
    t =np.array([dt * n for n in range(len(images))])

    return dt, t, images, metadata


def select_data_sets_to_process(data_directory: str, set_pattern: str,
                               output_file_name: str, reevaluate: bool) \
                               -> list[str]:
    """selects all data set directories in the data directory that match a 
       certain pattern.
    
    Args:
        data_directory (str): give the root directory of all data
        set_pattern (str): specify set name pattern for data set selection
        output_file_name (str): specify name of csv-data file
        reevaluate (bool): If set to false, sets that already have output file
                           will be omitted.

    Returns:
        _type_: _description_
    """
    data_sets = os.listdir(data_directory)
    data_sets_to_process = []

    for data_set in data_sets:
        data_set_path = os.path.join(data_directory, data_set)
        if re.match(set_pattern, data_set):
            uneval_path = os.path.join(data_set_path, "unevaluable")
            unevaluable = os.path.exists(uneval_path)

            output_file_path = os.path.join(data_set_path, output_file_name)
            output_file_exists = os.path.exists(output_file_path)

            if not unevaluable and (not output_file_exists or reevaluate):
                data_sets_to_process.append(data_set)
    
    return data_sets_to_process


def load_example_sets():
    """Downloads the example data sets provided with the code.
    """
    dir_name = "example_sets"

    if os.path.exists(dir_name):
        print("Example sets already downloaded and unpacked!")
        return
    
    file_mirror = "https://s.kit.edu/dropimpactexamplesets"
    zip_name = "example_sets.zip"


    msg = "WARNING: This routine will download ~70MB of image data. "
    msg += "If you want to proceed, input 'y'.\n"
    msg += "Alternatively, you can download and unpack the files yourself"
    msg += f"here: {file_mirror}"

    confirm = input(msg)

    if confirm == 'y':
        print("Downloading example data sets.")
        urllib.request.urlretrieve(file_mirror, zip_name)
        
        print("Unpacking example data sets.")
        with zipfile.ZipFile(zip_name, 'r') as f:
            f.extractall(".")
        
        os.remove(zip_name)
        print("Finished.")

