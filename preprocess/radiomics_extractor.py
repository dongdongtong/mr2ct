"""Just extract radiomics features of the 3D image and the corresponding segmentation mask.
Created by dsd in 2022.11.24.
NOTICE: We need not only extract the radiomics of each subject, but also extract the train_min, train_max radiomics for the training split.
"""

import os
import collections

from radiomics import featureextractor


def radiomics_feats_extractor(sitk_img, sitk_seg, extract_setting_yaml="preprocess/radiomics_ct.yaml"):
    extractor = featureextractor.RadiomicsFeatureExtractor(extract_setting_yaml)
    feature_dict = {}
    try:
        feature = extractor.execute(sitk_img, sitk_seg)
        feature_sorted = collections.OrderedDict(sorted(feature.items()))
        for feature_name in feature_sorted.keys():
            if 'diagnostics' not in feature_name:
                feature_dict[feature_name] = feature_sorted[feature_name]
    except Exception as e:
        print(e)

    if len(feature_dict) < 1379:
        print('Abnormal Input Size:', len(feature_dict))
        return None
    else:
        return feature_dict