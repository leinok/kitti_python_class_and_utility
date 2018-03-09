# kitti_velodyne.py: SAICUSA Perception 
# Description: this file process kitti lidar data which mainly include: read data, convert to bird view and so on.
# Mode detail can be found at the comment respectively.
from __future__ import absolute_import, division
import numpy as np
import cv2
import os
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
import time
import pdb

# read velodyne bin file from KITTI dataset, 000000.bin -- 007480.bin in training
def readLidarBin(idx, bin_folder = '/mnt/data_0/kitti/training/velodyne/'):
    bin_name = os.path.join(bin_folder, '%06d.bin'%idx)
    if not os.path.isfile(bin_name):
        print("{} does not exist!".format(bin_name))
        return None
    bin_data = np.fromfile(bin_name, dtype = np.float32).reshape(-1, 4)
    return bin_data

# keep the points inside the given boundary 
def filterPoint(bin_data, boundary = [1.0, 21.0, -20, 20, -2.0, 0.4]):
    within_index = (bin_data[:, 0] >= boundary[0]) & (bin_data[:, 0] < boundary[1]) \
		 & (bin_data[:, 1] >= boundary[2]) & (bin_data[:, 1] < boundary[3]) \
		 & (bin_data[:, 2] >= boundary[4]) & (bin_data[:, 2] < boundary[5])
    return within_index

# Get rid of the intensity layer.
# Features in (M + 1) channels, namely, M height maps, density and intensity
# M height maps: maximum height of points in cell, 
# density: number of points in each cell, normalized as min(1.0, log(N+1) / log(64))
# intensity: reflectance of point with maximum height in cell value in (0, 1)
# Parameter:
# -- bin_data: numpy data read from 00xxxx.bin
# -- M: as described above
# -- boundary [xmin, xmax, ymin, ymax, zmin, zmax]
# -- resolution: [x_resolution, y_resolution]
def generateBvMap(bin_data, M = 1, boundary = [1.0, 21.0, -20, 20, -2.0, 0.4], resolution = [0.1, 0.1]):
    if boundary[1] <= boundary[0] or boundary[3] <= boundary[2] or boundary[5] <= boundary[4]:
        raise ValueError('boundary range is nor reasonable.')	 
    img_width = int(np.ceil((boundary[1] - boundary[0]) / resolution[0])) 
    img_height = int(np.ceil((boundary[3] - boundary[2]) / resolution[1]))
    
    bv_map = np.zeros((img_width, img_height, M + 1))
    # Height Normalization
    bin_data[:, 2] = (bin_data[:, 2] - boundary[4]) / (boundary[5] - boundary[4])
    norm_slice_height = 1.0 / M
 

    num_points = bin_data.shape[0]
    for pidx in xrange(num_points):
        (x, y, z, r) = bin_data[pidx]
        x_bv = int(np.floor((x - boundary[0]) / resolution[0]))
        y_bv = int(np.floor((y - boundary[2]) / resolution[1]))
        z_bv = int(np.floor( z / norm_slice_height))

        if z > bv_map[x_bv][y_bv][z_bv]:
            bv_map[x_bv][y_bv][z_bv] = z

	# Density
        bv_map[x_bv][y_bv][M] = bv_map[x_bv][y_bv][M] + 1   
    

    # Normalize density 
    bv_map[:,:,M] = np.log(bv_map[:,:,M] + 1) / np.log(64)
    bv_map[:,:,M] = np.clip(bv_map[:,:,M], 0., 1.)
    return bv_map
    # This is just a temporary resort to keep the dimension same with the annotations
    # return np.transpose(bv_map, (1, 0, 2))

def rescaleTo255(data_array, min_value = 0., max_value = 1.):
    st = time.time()
    rescaled = (255.0 / max_value  * (data_array - min_value ))
    return rescaled
    
def saveBvMap(idx, M, theta, bin_folder, save_directory):
#    boundary = [1.0, 21.0, -20, 20, -2.0, 0.4]
    boundary = [0.0, 60.0, -30, 30, -2.0, 0.5]
    feat_format = "npy"
    bin_data = readLidarBin(idx, bin_folder = bin_folder)
    if theta != 0:
        bin_data = rotateLidar(bin_data, theta)
    bin_data = bin_data[filterPoint(bin_data, boundary = boundary), :]
    bv_map = generateBvMap(bin_data, M = M, boundary = boundary)
    # Each channel make sense for range [0, 1]
    rescaled = rescaleTo255(bv_map).astype(np.uint8)

    if feat_format == "npy":
        save_name = os.path.join(save_directory, 'bbox2D_01%04d.npy'%idx)
        np.save(save_name, rescaled)
    elif feat_format == "png":
        save_name = os.path.join(save_directory, 'bbox2D_01%04d.png'%idx)
        im = Image.fromarray(rescaled)
        im.save(save_name)
    print(idx)

# Generate front view map
# -- v_res / h_res: vertical and horizontal resolution
# -- v_fov (min_negative_angle, max_positive_angle)
def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) == 2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 100               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_axis_bgcolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        fig.show()

def rotateLidar(bin_data, theta = 0):
    theta = theta / 180. * np.pi
    num_points = bin_data.shape[0]
    for pidx in xrange(num_points):
        (x, y, z, r) = bin_data[pidx]
        bin_data[pidx] = [y * np.sin(theta) + x * np.cos(theta),
			  y * np.cos(theta) - x * np.sin(theta),
			  z, r]
	#bin_data[pidx] = [x * np.cos(theta) - y * np.sin(theta),
	# 	    x * np.sin(theta) + y * np.cos(theta),
	# 	    z, r]   
    return bin_data

def multiThreadsGenerateSequenceBvMap():
    M = 5
    theta = -45 
    bin_folder = '/mnt/data_0/kitti/training/velodyne/'
#    save_directory = os.path.join("/mnt/data_0/kitti/testing/", 'new_bv_feat{}'.format(M + 2))
    save_directory = os.path.join('/mnt/data_0/kitti/training/', "rotate-45_nointensity_new_bv_feat{}".format(M+1))   
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    my_pool = mp.Pool(processes = 40)
    my_pool.map(partial(saveBvMap, M = M, theta = theta, bin_folder = bin_folder, save_directory = save_directory), range(7481))
    my_pool.close()
    my_pool.join()
    print("-------------Task Completed-------------------")

if __name__ == "__main__":
# ============================================================================
    # Generate BV map by parallel computing
    choice_num = 1
    if choice_num == 1:
        multiThreadsGenerateSequenceBvMap()

# ============================================================================
    elif choice_num == 2: 
        lidar = readLidarBin(0) 
        HRES = 0.5         # horizontal resolution (assuming 20Hz setting)
        VRES = 0.4          # vertical res
        VFOV = (-24.9, 2.0) # Field of view (-ve, +ve) along vertical axis
        Y_FUDGE = 5         # y fudge factor for velodyne HDL 64E

        lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth",
                       saveto="./lidar_depth.png", y_fudge=Y_FUDGE)

        lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="height",
                       saveto="./lidar_height.png", y_fudge=Y_FUDGE)

        lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="reflectance", 
		       saveto="./lidar_reflectance.png", y_fudge=Y_FUDGE)

# =============================================================================
