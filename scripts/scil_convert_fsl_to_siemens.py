#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert bval/bvec FSL style to Siemens format.
"""

import argparse
import logging
import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_gradients_filenames_valid,
                             assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg, add_verbose_arg)
from scilpy.utils.bvec_bval_tools import fsl2siemens


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('fsl_bval',
                   help='Path to FSL b-value file (.bval).')

    p.add_argument('fsl_bvec',
                   help='Path to FSL gradient directions file (.bvec).')

    p.add_argument('siemens_dvs',
                   help='Output path for siemens gradient table file (.dvs).')

    p.add_argument('--reference',
                   help='Reference Nifti image to perform directions '
                        'rotation. FSL format is in image space, while '
                        'Siemens format is in scanner space. If supplied, will '
                        'invert the transformation (rotation only) from the '
                        'reference image to get the b-vectors in scanner space')

    p.add_argument('--b_nominal', type=int, default=None,
                   help="B-value that will be used with the siemens gradient "
                        "table at the scanner. If None supplied, the maximal "
                        "b-value from the .bval file will be used. Each "
                        "direction's effective b-value is computed using "
                        "b-nominal as reference, scaled by it's norm, using "
                        "the specified normalization strategy.")

    p.add_argument(
        '--norm_strat', default="none", choices=["none", "unity", "maximum"],
        help="Normalization strategy to get the effective b-value")

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_gradients_filenames_valid(parser, [args.fsl_bval, args.fsl_bvec],
                                     'fsl')

    # TODO: re-add and decomment in utils.py + have a better error message
    assert_gradients_filenames_valid(parser, args.siemens_dvs, 'siemens')
    assert_inputs_exist(
        parser, [args.fsl_bval, args.fsl_bvec], [args.reference])
    assert_outputs_exist(parser, args, args.siemens_dvs)

    if args.reference:
        img = nib.load(args.reference)
        affine = img.as_reoriented([[0, -1], [1, -1], [2, 1]]).affine
        rotation = nib.affines.rescale_affine(
            affine, img.shape[:3], (1, 1, 1))[:3, :3]
    else:
        rotation = np.eye(3)

    fsl2siemens(
        args.fsl_bval, args.fsl_bvec, args.siemens_dvs, rotation,
        args.b_nominal, args.norm_strat)


if __name__ == "__main__":
    main()
