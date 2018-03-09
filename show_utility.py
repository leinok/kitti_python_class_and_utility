import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import cv2
import argparse
import sys

def showPC(point_cloud, bbox3D_label=None):
    """
    Show point cloud using pyqtopengl,
    bbox3D_label is a ndarray(n, 7) --> (x, y, z, l, w, h, yaw_angle)
    """
    app = QtGui.QApplication([])
    pg.mkQApp()
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    view_widget = gl.GLViewWidget()
    view_widget.show()
    xgrid = gl.GLGridItem()
    ygrid = gl.GLGridItem()
    zgrid = gl.GLGridItem()
    view_widget.addItem(xgrid)
    view_widget.addItem(ygrid)
    view_widget.addItem(zgrid)

    line = gl.GLLinePlotItem()
    view_widget.addItem(line)

    xgrid.rotate(90, 0, 1, 0)
    ygrid.rotate(90, 1, 0, 0)
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    selected_area = np.where((y > 0))[0]
    specified_area = np.where((z > -2.0) & (z < 2.5) & (x > 30) & (x < 35))[0]
    #n, bins, patches = plt.hist(point_cloud[selected_area, 3], 10, normed=1, facecolor='green', alpha=0.75)
    #plt.show()
    scatter_plot = gl.GLScatterPlotItem(pos = point_cloud[:, :3], color = pg.glColor('g'), size = 0.1)
    view_widget.addItem(scatter_plot)
    if bbox3D_label is not None:
        #mkPen('y', width = 3, stype=QtCore.Qt.DashLine)
        bbox3D_corners = center_to_corner_box3d(bbox3D_label)
        draw3DBox(view_widget, bbox3D_corners)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        [l, w, h] = box[3:6]
        yaw = box[-1]

        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]])

        # re-create 3D bounding box in velodyne coordinate system
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret

def draw3DBox(w, bbox3D_corners):
    edge_list = [[0, 1], [1, 2], [2, 3], [3, 0], 
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
    color = pg.glColor('r')
    for box_idx in xrange(bbox3D_corners.shape[0]):
        cur_box = bbox3D_corners[box_idx, :, :]
        for edge_idx in edge_list:
            pts = cur_box[edge_idx, :]
            plt = gl.GLLinePlotItem(pos = pts, color = color)
            w.addItem(plt)

def featureMap(feature_map): 
    """
    Show nth level feature map, height map + density map
    """
    f, axarr = plt.subplots(2, 3)
    plt.suptitle('Feature Map')
    for i in range(0, 2):
        for j in range(0, 3):
            ith_height = i*3+j
            if i == 0 and j == 0:
                im = axarr[0, 0].imshow(feature_map[ith_height, :, :], vmin = 0, vmax = 255)
            else:
                axarr[i, j].imshow(feature_map[ith_height, :, :], vmin = 0, vmax = 255)
            if ith_height < 5:
                axarr[i, j].set_xlabel('The {}_th height map'.format(i*3+j+1))
            else:
                axarr[i, j].set_xlabel('The density map')

    cbaxes = f.add_axes([0.92, 0.11, 0.02, 0.75])
    cb = plt.colorbar(im, cax = cbaxes)
    plt.show()

def showImgWithPosition(img):
    """
    Show 2D image with their position index
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ax.imshow(img, cmap=cm.jet, interpolation='nearest')
    n_rows, n_cols= img.shape
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < n_cols and row >= 0 and row < n_rows:
            z = img[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)
    ax.format_coord = format_coord
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility Test Options')
    parser.add_argument('--use', type=str, nargs='?', default='show_lidar',
                                        help='Show_lidar/Show_featuremap Option')
    parser.add_argument('--pc', type=str, nargs='?', default='/mnt/data_0/kitti/training/velodyne/000000.bin',
                                        help='The path of point cloud')

    args = parser.parse_args()
    print(args.use)
    if args.use == 'show_lidar':
        point_cloud = np.fromfile(args.pc, np.float32).reshape(-1, 4)
        showPC(point_cloud)
    elif args.use == 'show_featuremap':
        bin_name = "/mnt/raid1/Research/lidar-processing/data.bin"
        bin_data = np.fromfile(bin_name, dtype = np.uint8).reshape(6, 600, 600)
        featureMap(bin_data)
