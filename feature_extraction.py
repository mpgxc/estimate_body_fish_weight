#!/usr/bin/env python
# coding: utf-8

from glob import glob
from tqdm import tqdm
import pandas as pd
from skimage.io import imread
from skimage.measure import label, regionprops_table
from skimage.transform import resize

ALL_REGION_PROPS_PROPERTIES = [
    "area",
    "bbox",
    "bbox_area",
    "centroid",
    "convex_area",
    "eccentricity",
    "equivalent_diameter",
    "euler_number",
    "extent",
    "filled_area",
    "label",
    "major_axis_length",
    "minor_axis_length",
    "moments",
    "moments_central",
    "orientation",
    "perimeter",
    "feret_diameter_max",
    "solidity",
]

from joblib import Parallel, delayed

def extract_features_from_images(
    images_arr=None,
    images_path=None,
    output_file_name="features.csv",
    resize_shape=(224, 224),
    properties=ALL_REGION_PROPS_PROPERTIES,
):
    """
    Extracts region properties from binary masks obtained from images in a given directory.

    Args:
        images_path (str): Path to directory containing the images to be processed.
        output_file_name (str): Name of the output file where the extracted properties will be saved.
        resize_shape (tuple): Tuple of two integers specifying the desired shape (height, width) for resizing the images.
        properties (list): List of strings containing the names of the region properties to be extracted.

    Returns:
        None

    Raises:
        None

    The function iterates over all image files in the given directory, resizes them to the desired shape,
    applies a binarization operation to obtain a binary mask, and then extracts the region properties
    specified in the 'properties' argument using the regionprops_table function from the skimage library.
    The output properties are then saved to a CSV file with the given name.
    If an error occurs while processing an image, the function skips that image and prints an error message.
    """

    df_output_props = pd.DataFrame()
    
    if len(images_arr):
        for (path, mask) in tqdm(zip(images_path, images_arr)):
            try:

                props = pd.DataFrame(regionprops_table(
                    label(mask), 
                    properties=properties)
                )
                
                props["label"] = path.split("/")[-1]
                
        
                df_output_props = df_output_props.append(
                    props, 
                    ignore_index=True,
                    verify_integrity=True
                )

            except Exception as image_error:
                print("Erro na imagem: ", image_error)
                continue
        

                
        df_output_props.to_csv(output_file_name, index=None)

    else:
        
        def make_row(image_path, df_output_props):
            
            image = imread(image_path)

            mask = resize(image, resize_shape, anti_aliasing=True) > 0.5

            props = pd.DataFrame(regionprops_table(
                label(mask),
                properties=properties)
            )

            props["label"] = image_path.split("/")[-1]

            return props


        try:

            r = Parallel(n_jobs=-2)(delayed(make_row)(image_path, df_output_props) for image_path in tqdm(glob(f"{images_path}/*")))
            
            for props in r:
                df_output_props = df_output_props.append(
                    props, ignore_index=True, verify_integrity=True
                )


        except Exception as image_error:
            print("Erro na imagem:", image_error)


        df_output_props.to_csv(output_file_name, index=None)
