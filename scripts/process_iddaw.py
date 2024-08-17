import os
import sys
import glob
import time
import shutil
import cv2
import random
import json
import argparse

import numpy as np

from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool


SCENES_DICT = { "FOG": 0, "LOWLIGHT": 1, "RAIN": 2, "SNOW": 3 }
SCENES_LIST = [ "FOG", "LOWLIGHT", "RAIN", "SNOW" ]

SPLIT_DICT = { "train": 0, "val": 1, "test": 2 }
SPLIT_LIST = [ "train", "val", "test" ]


def get_args():
    parser = argparse.ArgumentParser(description='Process IDDAW Dataset')

    parser.add_argument('--dir_iddaw', type=str, default="", required=True, metavar="/path/to/iddaw/dataset",
                        help='Path to IDDAW dataset')
    parser.add_argument('--calc_stats', action='store_true',
                        help='Calculate IDDAW dataset statistics')
    parser.add_argument('--convert', type=str, choices=["flat", "scenes"],
                        help='Convert to MMSeg data formats')
    parser.add_argument('--dir_mmseg', type=str, default=".", metavar="/path/to/output/directory",
                        help='Path to output directory')
    parser.add_argument('--num_workers', type=int, default=1, metavar=f"[1,...,{os.cpu_count()}]",
                        help='Number of worker threads for multiprocessing')

    args = parser.parse_args()
    return args


def json_write(file_path, data):
    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return obj.item()
            return json.JSONEncoder.default(self, obj)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyArrayEncoder)


def calc_stats(data_dir_iddaw):
    data_files_list = glob.glob(os.path.join(data_dir_iddaw, "*", "*", "rgb", "*", "*.png"))
    random.shuffle(data_files_list)

    stats = {
        sp: {
            sc: {
                'count': 0,
                'mean': np.zeros(3),
                'var': np.zeros(3)
            } for sc in SCENES_LIST
        } for sp in SPLIT_LIST
    }

    ids = { sp: { sc: set() for sc in SCENES_LIST } for sp in SPLIT_LIST}

    for fp in tqdm(data_files_list, desc="Calculating IDDAW Statistics"):
        k = fp.rsplit('/', 5)[1:]
        img = cv2.imread(fp).astype(np.float64)

        ids[k[0]][k[1]].add(int(k[3]))

        # [todo] optimize
        stats[k[0]][k[1]]['count'] += 1
        stats[k[0]][k[1]]['mean'] += np.mean(img, axis=(0, 1))
        stats[k[0]][k[1]]['var'] += np.mean(img**2, axis=(0, 1))

    ids = { sp: { sc: sorted(ids[sp][sc]) for sc in SCENES_LIST } for sp in SPLIT_LIST}

    json_write("ids.json", ids)

    stats = {
        sp: {
            sc: {
                'count': stats[sp][sc]['count'],
                'mean': np.divide(stats[sp][sc]['mean'][::-1], stats[sp][sc]['count']),
                'var': np.divide(stats[sp][sc]['var'][::-1], stats[sp][sc]['count']) - (np.divide(stats[sp][sc]['mean'][::-1], stats[sp][sc]['count']) ** 2)
            } for sc in SCENES_LIST
        } for sp in SPLIT_LIST
    }

    tots = {
        sp : {
            'count': np.sum([stats[sp][sc]['count'] for sc in SCENES_LIST]),
            'mean': np.sum([stats[sp][sc]['mean'] for sc in SCENES_LIST], axis=0) / len(SCENES_LIST),
            'stddev': np.sqrt(np.sum([stats[sp][sc]['var'] for sc in SCENES_LIST], axis=0) / len(SCENES_LIST))
        } for sp in SPLIT_LIST
    }

    json_write("tots.json", tots)

    stats = {
        sp: {
            sc: {
                'count': stats[sp][sc]['count'],
                'mean': stats[sp][sc]['mean'],
                'stddev': np.sqrt(stats[sp][sc]['var'])
            } for sc in SCENES_LIST
        } for sp in SPLIT_LIST
    }

    json_write("stats.json", stats)


def copy_image(paths):
    src_path, dst_path = paths

    try:
        img = Image.open(src_path)
    except:
        pass

    try:
        img.save(dst_path)
    except:
        pass


def iddaw_to_mmseg(dir_iddaw, dir_mmseg, num_workers):
    for dt in ["img_dir", "ann_dir"]:
        for ds in ["train", "val"]:
            dp = os.path.join(dir_mmseg, dt, ds)
            if not os.path.exists(dp):
                os.makedirs(dp)

    iddaw_files_list = []
    iddaw_files_list.extend(sorted(glob.glob(os.path.join(dir_iddaw, "*", "*", "rgb", "*", "*_rgb.png"))))
    iddaw_files_list.extend(sorted(glob.glob(os.path.join(dir_iddaw, "*", "*", "gtSeg", "*", "*_label.png"))))

    mmseg_files_list = []
    for ifp in iddaw_files_list:
        toks = ifp.rsplit('/', 5)[1:]

        if toks[2] == "rgb":
            mfp = os.path.join(dir_mmseg, "img_dir", toks[0], "_".join([toks[3], toks[4].replace("_rgb", "")]))
        elif toks[2] == "gtSeg":
            mfp = os.path.join(dir_mmseg, "ann_dir", toks[0], "_".join([toks[3], toks[4].replace("_label", "")]))

        mmseg_files_list.append(mfp)

    pool = Pool(num_workers)
    res = list(
        tqdm(
            pool.imap_unordered(
                copy_image,
                list(zip(iddaw_files_list, mmseg_files_list))
            ),
            total=len(iddaw_files_list)
        )
    )

    pool.close()
    pool.join()


if __name__ == '__main__':
    args = get_args()

    if args.calc_stats:
        calc_stats(args.dir_iddaw)

    if args.convert is not None:
        dir_mmseg = os.path.join(args.dir_mmseg, '_'.join(["iddaw", args.convert]))
        if args.convert == "flat":
            iddaw_to_mmseg(args.dir_iddaw, dir_mmseg, args.num_workers)
        elif args.convert == "scenes":
            pass
