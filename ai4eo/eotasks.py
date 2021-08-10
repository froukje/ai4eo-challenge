#!/usr/bin/env python
# Built-in modules
import os
import json
import datetime as dt
from typing import Tuple, List

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm

# Module for GeoDB
from xcube_geodb.core.geodb import GeoDBClient

# Imports from eo-learn and sentinelhub-py
from sentinelhub import CRS, BBox, SHConfig, DataCollection

from eolearn.core import (FeatureType,
                          EOPatch, 
                          EOTask, 
                          LinearWorkflow, 
                          EOExecutor, 
                          LoadTask,
                          SaveTask)
from eolearn.io import GeoDBVectorImportTask, SentinelHubInputTask
from eolearn.geometry import VectorToRaster

# Visualisation utilities from utils.py
from utils import get_extent, md5_encode_files

class SamplePatchletsTask(EOTask):
    '''
    Sample patchlets from EOTask
    '''

    SCALE_FACTOR = 4 # do not change

    def __init__(self, s2_patchlet_size: int, num_samples: int, random_mode: bool):
        """ Set-up of task 
        
        :param s2_patchlet_size: Size in pixels of resulting patchlet
        :param num_samples: Number of patchlets to sample
        """
        self.s2_patchlet_size = s2_patchlet_size
        self.num_samples = num_samples
        self.random_mode = random_mode

    def _calculate_sampled_bbox(self, bbox: BBox, r: int, c: int, s: int,
                                resolution: float) -> BBox:
        """ Calculate bounding box of smaller patchlets """
        return BBox(((bbox.min_x + resolution * c,  bbox.max_y - resolution * (r + s)),
                     (bbox.min_x + resolution * (c + s), bbox.max_y - resolution * r)),
                    bbox.crs)

    def _sample_s2(self, eop: EOPatch, row: int, col: int, size: int, 
                   resolution: float = 10):
        """ Randomly sample a patchlet from the EOPatch """
        # create a new eopatch for each sub-sample
        sampled_eop = EOPatch(timestamp=eop.timestamp, 
                              scalar=eop.scalar, 
                              meta_info=eop.meta_info)
        
        # sample S2-related arrays
        features = eop.get_feature_list()
        s2_features = [feature for feature in features 
                       if isinstance(feature, tuple) and 
                       (feature[0].is_spatial() and feature[0].is_time_dependent())]
        
        for feature in s2_features:
            sampled_eop[feature] = eop[feature][:, row:row + size, col:col + size, :]
        
        # calculate BBox for new sub-sample
        sampled_eop.bbox = self._calculate_sampled_bbox(eop.bbox, 
                                                        r=row, c=col, s=size, 
                                                        resolution=resolution)
        sampled_eop.meta_info['size_x'] = size
        sampled_eop.meta_info['size_y'] = size
        
        # sample from target maps, beware of `4x` scale factor
        target_features = eop.get_feature(FeatureType.MASK_TIMELESS).keys()
        
        for feat_name in target_features:
            sampled_eop.mask_timeless[feat_name] = \
            eop.mask_timeless[feat_name][self.SCALE_FACTOR*row:self.SCALE_FACTOR*row + self.SCALE_FACTOR*size, 
                                         self.SCALE_FACTOR*col:self.SCALE_FACTOR*col + self.SCALE_FACTOR*size]
        
        return sampled_eop

    def execute(self, eopatch_s2: EOPatch, buffer: int=0,  seed: int=42, random_mode: bool=1) -> List[EOPatch]:
        """ Sample a number of patchlets from the larger EOPatch. 
        
        :param eopatch_s2: EOPatch from which patchlets are sampled
        :param buffer: Do not sample in a given buffer at the edges of the EOPatch
        :param seed: Seed to initialise the pseudo-random number generator
        :param random_mode: Select the upper left corner at random (default: True)
        """
        _, n_rows, n_cols, _ = eopatch_s2.data['BANDS'].shape
        np.random.seed(seed)
        eops_out = []
        
        if not self.random_mode:
            max_per_row = n_rows // self.s2_patchlet_size
            max_per_col = n_cols // self.s2_patchlet_size
        
        # random sampling of upper-left corner. Added: Change this for non-overlapping patchlets
        for patchlet_num in range(0, self.num_samples):
            if self.random_mode:
                row = np.random.randint(buffer, n_rows - self.s2_patchlet_size - buffer)
                col = np.random.randint(buffer, n_cols - self.s2_patchlet_size - buffer)
            else:
                row = (buffer + patchlet_num // int(np.floor((n_rows - buffer) / self.s2_patchlet_size)) * self.s2_patchlet_size)
                col = buffer + (patchlet_num * self.s2_patchlet_size) % (n_cols - buffer - self.s2_patchlet_size)
                
                row = (patchlet_num // max_per_row) * self.s2_patchlet_size
                col = (patchlet_num % max_per_col) * self.s2_patchlet_size
                
                
            sampled_s2 = self._sample_s2(eopatch_s2, row, col, self.s2_patchlet_size)
            eops_out.append(sampled_s2)

        return eops_out