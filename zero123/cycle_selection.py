import numpy as np
import torch
import json
import argparse
import shutil
import os

def merge_selected_views(base_dir, view_list, idx_list):
    merged_dir = os.path.join(base_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)

    merged_idx = 0
    meta = dict(frames=[])

    # reference view
    shutil.copy(os.path.join(base_dir, 'raw_img.png'), os.path.join(merged_dir, '{:05d}.png'.format(merged_idx)))
    with open(os.path.join(base_dir, 'raw_transforms.json')) as f:
        this_meta = json.load(f)
    c2w = this_meta['transform_matrix']
    meta['frames'].append(dict(file_path='merged/{:05d}.png'.format(merged_idx), transform_matrix=c2w))
    merged_idx += 1

    # zero123 views
    for view_idx, img_idx in zip(view_list, idx_list):
        img_file = os.path.join(base_dir, 'view_{:03d}/img_{:02d}.png'.format(view_idx, img_idx))
        shutil.copy(img_file, os.path.join(merged_dir, '{:05d}.png'.format(merged_idx)))

        with open(os.path.join(base_dir, 'view_{:03d}/transforms.json'.format(view_idx))) as f:
            this_meta = json.load(f)
        c2w = this_meta['frames'][img_idx]['transform_matrix']

        meta['frames'].append(dict(file_path='merged/{:05d}.png'.format(merged_idx), transform_matrix=c2w))
        merged_idx += 1

    with open(os.path.join(merged_dir, 'transforms.json'), 'w') as f:
        json.dump(meta, f)

def merge_get3d_views(base_dir, view_list, idx_list):
    merged_dir = os.path.join(base_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.mkdir(merged_dir)

    merged_idx = 0

    # zero123 views
    for view_idx, img_idx in zip(view_list, idx_list):
        img_file = os.path.join(base_dir, 'view_{:03d}/img_{:02d}.png'.format(view_idx, img_idx))
        shutil.copy(img_file, os.path.join(merged_dir, '{:05d}.png'.format(merged_idx)))
        merged_idx += 1


if __name__ == '__main__':
    # merge_selected_views('outputs/demo_minion3', np.arange(39).tolist(), np.zeros(40, dtype=int).tolist())
    merge_get3d_views('outputs/demo_minion_dense', np.arange(240).tolist(), np.zeros(240, dtype=int).tolist())
