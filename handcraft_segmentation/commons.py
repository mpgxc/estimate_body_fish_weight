import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import filters
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import statistics
from skimage import filters, measure, morphology, exposure, feature
import skimage
import numpy as np

from glob import glob as show_dir_files
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from sklearn.metrics import pairwise_distances_argmin
from skimage import filters, measure, morphology, exposure, feature
import statistics


def check_colision_border(mask):

    x, *_ = mask.shape

    left = mask[:1, ].flatten()
    right = mask[x - 1: x, ].flatten()
    top = mask[:, : 1].flatten()
    bottom = mask[:, x - 1: x].flatten()

    borders_flatten = [left, right, top, bottom]

    if np.concatenate(borders_flatten).sum():
        return True

    return False

def binarize_image(arr):
    return arr > filters.threshold_triangle(arr)

def plot(arr_images=[], grid=(2, 2), cmap="inferno"):

    fig = plt.figure(figsize=(20, 10))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=grid,
                     axes_pad=0.1)

    for ax, img in zip(grid, arr_images):
        ax.imshow(img, cmap)
        ax.axis('off')

    plt.show()
    
    
def load_images_from_directory_resize(arr_paths, is_gray=False, dim=(256, 256)):

    arr_images = []

    for img_path in tqdm(arr_paths):
        image = imread(img_path)
        image = resize(image, dim, anti_aliasing=True)
        
        if is_gray: 
            arr_images.append(rgb2gray(image))
        else: 
            arr_images.append(image)
            
    return np.asarray(arr_images)

  
def load_images_from_directory(arr_paths, is_gray=False):

    arr_images = []

    for img_path in tqdm(arr_paths):
        image = imread(img_path)

        if is_gray: 
            arr_images.append(rgb2gray(image))
        else: 
            arr_images.append(image)
            
    return np.asarray(arr_images)
  
    
def auto_invert_image_mask(arr):

    """
    Calcula os pixels da imagem e inverte os pixels da imagem caso os pixels True > False
    Isso é uma forma de garatir que as mascaras tenham sempre o fundo preto = 0 e o ROI = 1
    """

    img = arr.copy()

    if statistics.mode(img.flatten()):
        img = np.invert(img)

    return img


def find_bighest_cluster_area(clusters):
    regions = measure.regionprops(clusters)

    all_areas = map(lambda item: item.area, regions)

    return max(all_areas)


def find_bighest_cluster(img):
  
    clusters = auto_invert_image_mask(img)

    clusters = measure.label(clusters, background=0)

    cluster_size = find_bighest_cluster_area(clusters)

    return morphology.remove_small_objects(clusters.astype(bool),
                                         min_size=(cluster_size - 1),
                                         connectivity=8)


def find_roi(img):

    binary_img = binarize_image(exposure.equalize_hist(img))

    best_cluster = find_bighest_cluster(binary_img)

    merged = binarize_image(binary_img + best_cluster)

    return binarize_image(find_bighest_cluster(merged))


def smoothing_mask_edges(mask):
    return binarize_image(filters.gaussian(mask, sigma=0.5))