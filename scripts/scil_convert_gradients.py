#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Script to convert between gradients protocols files formats
'''

import argparse
import logging
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg)

from scilpy.utils.bvec_bval_tools import (GradientProtocol,
                                          SIEMENS_NORMALIZATION,
                                          SUPPORTED_FORMATS)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('target_format', choices=SUPPORTED_FORMATS,
                   help='File format of the output gradient protocol')
    p.add_argument('output_fname', help='Filename of the output protocol')

    pg = p.add_mutually_exclusive_group(required=True)
    pg.add_argument(
        '--fsl', nargs=2,
        help='Supply input protocol in FSL format : bval + bvec files ')
    pg.add_argument(
        '--mrtrix',
        help='Supply input protocol in mrtrix format : <protocol>.b file')
    pg.add_argument(
        '--siemens',
        help='Supply input protocol in Siemens format : <protocol>.dvs file')

    sg = p.add_argument_group("Siemens format options")
    sg.add_argument("--b_nominal", type=float,
                    help="Nominal b-value to consider when "
                         "reading/writing a Siemens file")
    sg.add_argument("--normalization",
                    choices=SIEMENS_NORMALIZATION,
                    default="none",
                    help="Normalization strategy to consider when "
                         "reading/writing a Siemens file")
    sg.add_argument(
        "--reference", help="Reference image for the input gradients. If "
                            "reading from Siemens, will apply the image "
                            "transform to output the directions in image "
                            "space. If writing to Siemens, will apply the "
                            "inverse transform to get the directions in "
                            "scanner space.")

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    ref_affine = np.eye(4)
    if args.reference:
        ref_affine = nib.load(args.reference).affine

    _protocol = None
    if args.fsl is not None:
        _protocol = GradientProtocol.from_fsl(*args.fsl)
    elif args.mrtrix is not None:
        _protocol = GradientProtocol.from_mrtrix(args.mrtrix)
    elif args.siemens is not None:
        _protocol = GradientProtocol.from_siemens(args.siemens, args.b_nominal,
                                                  ref_affine)
    else:
        parser.error("Invalid input gradient format specified. "
                     "Must be one of : {}".format(SUPPORTED_FORMATS))

    _protocol.save(args.target_format, args.output_fname,
                   b_nominal=args.b_nominal or 1.,
                   normalization=args.normalization,
                   ref_affine=ref_affine)


if __name__ == '__main__':
    main()
