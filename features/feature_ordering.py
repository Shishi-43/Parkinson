
import numpy as np


COLS = [
    'zcr', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff',
    'pitch_mean', 'pitch_std',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
    'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13'
]

def features_to_ordered_array(feat_dict):
    return np.array([feat_dict[c] for c in COLS]).reshape(1, -1)