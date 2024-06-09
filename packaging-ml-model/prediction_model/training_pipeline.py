import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# just to avoid the error - adding the path 
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

#second the solution
# Add project root to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.join(current_dir, '..')
# sys.path.append(project_root)
# print("Updated Python Path:", sys.path)


from prediction_model.config import config
from prediction_model.processing.data_handling import load_datasets, save_pipeline
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipeline



def perform_training():
    
    train_data=load_datasets(config.TRAIN_FILE)
    train_y=train_data[config.TARGET].map({"N":0,"Y":1})
    pipeline.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    save_pipeline(pipeline.classification_pipeline)

if __name__=='__main__':
    perform_training()