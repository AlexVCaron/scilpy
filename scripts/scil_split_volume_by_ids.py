#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a label image into multiple images where the name of the output images
is the id of the label (ex. 35.nii.gz, 36.nii.gz, ...). If the --range option
is not provided, all labels of the image are extracted.

IMPORTANT: your label image must be of an integer type.
"""

import argparse
import os

import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels, split_labels
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_labels',
                   help='Path of the input label file, '
                        'in a format supported by Nibabel.')
    p.add_argument('--out_dir', default='',
                   help='Put all ouptput images in a specific directory.')
    p.add_argument('--out_prefix',
                   help='Prefix to be used for each output image.')

    p.add_argument('-r', '--range', type=int, nargs=2, metavar='min max',
                   action='append',
                   help='Specifies a subset of labels to split, formatted as '
                        'min max. Ex: -r 3 4.')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    required = args.in_labels
    assert_inputs_exist(parser, required)

    label_img = nib.load(args.in_labels)
    label_img_data = get_data_as_labels(label_img)

    if args.range is not None:
        # Ex: From 173 to 175 = range(173, 176).
        label_indices = np.concatenate(
            [np.arange(r[0], r[1] + 1) for r in args.range])
    else:
        label_indices = np.unique(label_img_data)
    label_names = [str(i) for i in label_indices]

    output_filenames = []
    for label, name in zip(label_indices, label_names):
        if int(label) != 0:
            if args.out_prefix:
                output_filenames.append(os.path.join(
                    args.out_dir,
                    '{0}_{1}.nii.gz'.format(args.out_prefix, name)))
            else:
                output_filenames.append(os.path.join(
                    args.out_dir, '{0}.nii.gz'.format(name)))

    assert_output_dirs_exist_and_empty(parser, args, [], optional=args.out_dir)
    assert_outputs_exist(parser, args, output_filenames)

    # Extract the voxels that match the label and save them to a file.
    split_data = split_labels(label_img_data, label_indices)

    for i in range(len(label_indices)):
        if split_data[i] is not None:
            split_image = nib.Nifti1Image(split_data[i],
                                          label_img.affine,
                                          header=label_img.header)
            nib.save(split_image, output_filenames[i])


if __name__ == "__main__":
    main()
