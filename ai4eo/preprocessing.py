#!/usr/bin/env python
# Built-in modules
import os
import json
import datetime as dt
from typing import Tuple, List
import argparse

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd

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

from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.mask import AddValidDataMaskTask

# Visualisation utilities from utils.py
from utils import get_extent

# our own modules
import eotasks

def main(args):

    # --------------------
    # USEFUL DEFINITIONS
    # --------------------

    # band specification
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
    band_names = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'] # from starter notebook
    band_wavelength = [443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1610, 2190] # nm
    band_spatial_resolution = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20] # m

    # EOPatches for input
    eops_train = os.listdir(args.raw_data_dir)

    # --------------------
    # DEFINE THE WORKFLOW
    # --------------------

    # load eopatches from raw data
    load_eopatches = LoadTask(path=args.raw_data_dir)

    # compute reflectances
    compute_reflectances = eotasks.ComputeReflectances((FeatureType.DATA, 'BANDS'))

    # add the features: NDVI, NDWI, NDBI 
    ndvi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDVI'),
                                     [band_names.index('B08'), band_names.index('B04')])
    ndwi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDWI'),
                                     [band_names.index('B03'), band_names.index('B08')])
    ndbi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDBI'),
                                     [band_names.index('B11'), band_names.index('B08')])

    # validity masks
    # Validate pixels using SentinelHub's cloud detection mask and region of acquisition
    add_sh_validmask = AddValidDataMaskTask(eotasks.SentinelHubValidData(), 'IS_VALID')

    # Count the number of valid observations per pixel using valid data mask
    add_valid_count = eotasks.AddValidCountTask('IS_VALID', 'VALID_COUNT')

    # Filter out cloudy scenes
    valid_data_predicate = eotasks.ValidDataFractionPredicate(args.cloud_threshold)
    filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

    # subsample
    # move to data set loading - does not work with the workflow, produces a list
    #   sample_task = eotasks.SamplePatchletsTask(s2_patchlet_size=args.s2_length, 
    #                                             num_samples=15*15, 
    #                                             random_mode=args.s2_random)

    # add weight map
    add_weight = eotasks.AddWeightMapTask( (FeatureType.MASK_TIMELESS, 'CULTIVATED'), 
                                           (FeatureType.MASK_TIMELESS, 'NOT_DECLARED'), 
                                           (FeatureType.DATA_TIMELESS, 'WEIGHTS'))

    save_eopatches = SaveTask(args.target_dir, overwrite_permission=args.overwrite)

    # construct the graph
    workflow = LinearWorkflow(load_eopatches,
                              compute_reflectances,
                              ndvi,
                              ndwi,
                              ndbi,
                              add_sh_validmask,
                              add_valid_count,
                              filter_task,
                              #sample_task, 
                              add_weight,
                              save_eopatches,
                              )

    # --------------------
    # EXECUTE THE WORKFLOW
    # --------------------

    # Define additional parameters of the workflow
    execution_args = []

    # need to specify the files that should be loaded 
    # loop all available eopatches and add each to the argument list
    for eop_train in eops_train:
        eop_exec_args = {
                        load_eopatches: {'eopatch_folder': f'{eop_train}'},
                        save_eopatches: {'eopatch_folder': f'processed_{eop_train}'},
                        }
        execution_args.append(eop_exec_args)

    # Execute the workflow
    executor = EOExecutor(workflow, execution_args, save_logs=True)
    executor.run(workers=args.n_processes, multiprocess=True)

    executor.make_report()
    print('Report was saved to location: {}'.format(executor.get_report_filename()))

    failed_ids = executor.get_failed_executions()
    if failed_ids:
        raise RuntimeError(f'Execution failed EOPatches with IDs:\n{failed_ids}\n'
                           f'For more info check report at {executor.get_report_filename()}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', type=str, default='/work/shared_data/2021-ai4eo/eopatches/')
    parser.add_argument('--target-dir', type=str, default='/work/shared_data/2021-ai4eo/dev_data/default/')
    parser.add_argument('--n-processes', type=int, default=4, help='Processes for EOExecutor')
    #parser.add_argument('--n-valid-patches', type=int, default=10, help='Number of EOPatches selected for validation')
    #parser.add_argument('--s2-length', type=int, default=32, help='Cropped EOPatch samples side length')
    parser.add_argument('--cloud-threshold', type=float, default=0.9, help='threshold for valid data cloud mask')
    #parser.add_argument('--n-time-frames', type=int, default=1, help='Number of time frames in EOPatches')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output files')
    #parser.add_argument('--s2-random', action='store_true', 
    #                    help='Randomly select overlapping patches (else: systematically select non overlapping patches')

    args = parser.parse_args()

    print('\n*** begin args key / value ***')
    for key, value in vars(args).items():
        print(f'{key:20s}: {value}')
    print('*** end args key / value ***\n')

    main(args)
