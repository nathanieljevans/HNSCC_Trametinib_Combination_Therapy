'''
More info here: https://github.com/nathanieljevans/HNSCC_functional_data_pipeline 

author: nathaniel evans 
email: evansna@ohsu.edu
'''
####################################################################################################
####################################################################################################
REPO_DIR = r'/home/teddy/local/HNSCC_functional_data_pipeline'

PLATE_MAP_DIR = f'{REPO_DIR}/plate_maps/'

DATA_DIR = './../plate-data/'

OUTPUT_PATH = './../data/all_trem_combo_data.csv'
####################################################################################################
####################################################################################################

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import shutil
import time

import sys 
sys.path.append(REPO_DIR + '/python')
import HNSCC_analysis_pipeline_lib as lib


if __name__ == '__main__': 
    
    file_paths = os.listdir(DATA_DIR) 
    print('file paths:')
    _ = [print(f'\t{x}') for x in file_paths]
          
    _data_ls = []
    for fpath in file_paths:

        with open(f'./../data/processing.log', 'w') as f:
            f.write(f'Log output for processing of: {fpath} \n {"#"*20} \n')
            sys.stdout = f

            plate_data_path = DATA_DIR + fpath

            p = lib.panel(plate_path=plate_data_path, platemap_dir=PLATE_MAP_DIR, verbose=True, path_sep='/')
            lab_id = p.lab_id 
            p.map_data()
            p.data.head()
            p.normalize_cell_viability_by_negative_controls()
            p.set_floor()
            p.set_ceiling()
            _data_ls.append(p.data)
            
    all_data = pd.concat(_data_ls, axis=0)
    all_data = all_data[['lab_id','inhibitor','conc','cell_viab']]
    all_data.to_csv(OUTPUT_PATH)
            