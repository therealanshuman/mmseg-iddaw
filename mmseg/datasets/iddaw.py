"""IDDAWDataset Class Definition

This goes under mmsegmentation/mmseg/datasets
"""

import os.path as osp
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class IDDAWDataset(BaseSegDataset):

    METAINFO = dict(
        classes = (
            'road', 'parking', 'drivable fallback', 'sidewalk', 'rail track',
            'non-drivable fallback', 'person', 'animal', 'rider', 'motorcycle',
            'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'caravan',
            'trailer', 'train', 'vehicle fallback', 'curb', 'wall', 'fence',
            'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole',
            'polegroup', 'obs-str-bar-fallback', 'building', 'bridge',
            'tunnel', 'vegetation', 'sky', 'fallback background', 'unlabeled',
            'ego vehicle', 'rectification border', 'out of roi',
            'license plate'
        ),
        palette = [
            [0, 0, 3], [2, 1, 14], [6, 4, 27], [14, 8, 42], [22, 11, 57],
            [32, 12, 74], [43, 10, 86], [53, 9, 96], [65, 9, 103],
            [75, 12, 107], [87, 15, 109], [96, 19, 110], [106, 23, 110],
            [117, 27, 109], [126, 30, 108], [137, 34, 105], [147, 37, 103],
            [156, 41, 99], [167, 45, 95], [177, 49, 90], [187, 55, 84],
            [196, 60, 78], [204, 65, 72], [213, 73, 64], [220, 80, 57],
            [228, 90, 49], [233, 98, 42], [238, 108, 34], [243, 119, 25],
            [246, 129, 17], [249, 142, 8], [251, 153, 6], [251, 164, 10],
            [251, 177, 22], [250, 189, 35], [248, 203, 52], [245, 215, 69],
            [243, 226, 89], [241, 238, 116], [244, 247, 141]
        ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
