# -*- coding: utf-8 -*-

import logging
import os

import numpy as np


def save_gradient_sampling_siemens(
    points, shell_idx, bval_ratios, normalization, filename
):
    """
    Save table gradient (Siemens format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    bval_ratios: numpy.array
        Ratios of b-values to b_nominal.
    normalization: str
        Normalization strategy for Siemens gradient table reader.
    filename: str
        output file name.
    ------
    """
    with open(filename, 'w+') as f:
        f.write("[directions={}]\n".format(points.shape[1]))
        f.write("CoordinateSystem = xyz\n")
        f.write("Normalisation = {}\n".format(normalization))

        for idx in range(points.shape[1]):
            f.write('Vector[{}] = ({:.8f}, {:.8f}, {:.8f})\n'.format(
                idx, *(bval_ratios[shell_idx[idx]] * points[:, idx])))

    logging.info(
        'Gradient sampling saved in Siemens format as {}'.format(filename))


def save_gradient_sampling_mrtrix(points, shell_idx, bvals, filename):
    """
    Save table gradient (MRtrix format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    bvals: numpy.array
    filename: str
        output file name.
    ------
    """
    with open(filename, 'w') as f:
        for idx in range(points.shape[1]):
            f.write('{:.8f} {:.8f} {:.8f} {:}\n'.format(points[0, idx],
                                                        points[1, idx],
                                                        points[2, idx],
                                                        bvals[shell_idx[idx]]))

    logging.info(
        'Gradient sampling saved in MRtrix format as {}'.format(filename))


def save_gradient_sampling_fsl(points, shell_idx, bvals, filename_bval,
                               filename_bvec):
    """
    Save table gradient (FSL format)

    Parameters
    ----------
    points: numpy.array
        bvecs normalized to 1.
    shell_idx: numpy.array
        Shell index for bvecs in points.
    bvals: numpy.array
    filename_bval: str
        output bval filename.
    filename_bvec: str
        output bvec filename.
    ------
    """
    basename, ext = os.path.splitext(filename_bval)

    np.savetxt(filename_bvec, points, fmt='%.8f')
    np.savetxt(filename_bval,
               np.array([bvals[idx] for idx in shell_idx])[None, :], fmt='%.3f')

    logging.info('Gradient sampling saved in FSL format as {}'.format(
        basename + '{.bvec/.bval}'))
