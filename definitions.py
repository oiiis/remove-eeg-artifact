import os

DATASET_ROOT = '/Users/jiangwengyao/Desktop/A semi-simulated EEGEOG dataset for the comparison of EOG artifact rejection techniques'

PURE_EEG = os.path.join(DATASET_ROOT, 'Pure_Data.mat')

CONTAMINATED_EEG = os.path.join(DATASET_ROOT, 'Contaminated_Data.mat')

PURE_SUFFIX = '_resampled'

CON_SUFFIX = '_con'

RECORD_NUMBER = 54
