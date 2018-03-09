from __future__ import division, absolute_import

import os
import numpy as np

class BaseDataReader(object):
    def __init__(self, directory):
        """
        Initialize base data extractor
        """
        print("The dataset root directory is {}".format(directory))
        if not os.path.exists(directory):
            raise IOError("Could not find data directory '%s'" % directory)
        self._dir = directory

    def getDatasets(self):
        raise NotImplementedError("getDatasets() is not implemented in BaseDataReader")

    def getFrameInfo(self, frameID, dataset=None):
        raise NotImplementedError("getFrameInfo() is not implemented in BaseDataReader")
