import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
__C = edict()

cfg = __C

# The following 6 variables are designed for clip the LIDAR area
__C.left = 0
__C.right = 60
__C.bottom = -30
__C.top = 30
__C.low = -2
__C.high = 0.5

