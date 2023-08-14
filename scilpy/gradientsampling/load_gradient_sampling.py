# -*- coding: utf-8 -*-

import numpy as np


def load_gradient_sampling_fsl(fsl_bval_filename, fsl_bvec_filename):
    """
    Read FSL bval/bvec files to get the gradient protocol

    Parameters
    ----------
    fsl_bval_filename: str
        path to input fsl bval file.
    fsl_bvec_filename: str
        path to input fsl bvec file.
    Returns
    -------
    bvals: np.ndarray (K,)
        list of unique bvalues.
    shell_idx: np.ndarray (N,)
        indexes of the b-values in bvals associated to each gradient direction 
        in points.
    points: np.ndarray (3, N)
        list of gradient directions.
    """
    points = np.loadtxt(fsl_bvec_filename)

    shells = np.loadtxt(fsl_bval_filename)
    bvals = np.unique(shells)
    shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]

    return bvals, shell_idx, points


def load_gradient_sampling_mrtrix(mrtrix_filename):
    """
    Read Mrtrix .b file to get the gradient protocol

    Parameters
    ----------
    mrtrix_filename : str
        path to mrtrix encoding.b file.
    Returns
    -------
    bvals: np.ndarray (K,)
        list of unique bvalues.
    shell_idx: np.ndarray (N,)
        indexes of the b-values in bvals associated to each gradient direction 
        in points.
    points: np.ndarray (3, N)
        list of gradient directions.
    """
    mrtrix_b = np.loadtxt(mrtrix_filename)
    if not len(mrtrix_b.shape) == 2 or not mrtrix_b.shape[1] == 4:
        raise ValueError('mrtrix file must have 4 columns')

    points = np.array([mrtrix_b[:, 0], mrtrix_b[:, 1], mrtrix_b[:, 2]])
    shells = np.array(mrtrix_b[:, 3])

    bvals = np.unique(shells).tolist()
    shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]

    return bvals, shell_idx, points


def load_gradient_sampling_siemens(siemens_filename, b_nominal,
                                   ref_affine=np.eye(4)):
    """
    Read Siemens .dvs file to get the gradient protocol

    Parameters
    ----------
    siemens_filename : str
        path to siemens <encoding>.dvs file.
    b_nominal: float
        nominal b-value of the gradients protocol.
    ref_affine: np.ndarray (4, 4), optional
        reference affine transform to image. If available, will 
        be used to transform the directions in image space.
    Returns
    -------
    bvals: np.ndarray (K,)
        list of unique bvalues.
    shell_idx: np.ndarray (N,)
        indexes of the b-values in bvals associated to each gradient direction 
        in points.
    points: np.ndarray (3, N)
        list of gradient directions.
    """

    with open(siemens_filename) as siemens_dvs:
        normalization, n_dirs = None, None
        while normalization is None or n_dirs is None:
            line = siemens_dvs.readline()
            if line[0] == "#":
                continue
            if "directions" in line.lower():
                n_dirs = int(line.strip("[]").split("=")[1])
            if "normalization" in line.lower():
                normalization = line.split("=")[1].strip().lower()

        points = np.empty((3, n_dirs))
        shells = np.empty((n_dirs,))
        for line in siemens_dvs:
            if "vector" in line.lower():
                v, coords = line.split("=")
                v = int(v.split("[")[1].strip("] "))
                coords = eval(coords.strip())
                norm = np.linalg.norm(coords)
                points[:, v] = ref_affine[:3, :3] @ (coords / norm)
                shells[v] = norm * b_nominal

        bvals = np.unique(shells).tolist()
        shell_idx = [int(np.where(bval == bvals)[0]) for bval in shells]

        return bvals, shell_idx, points
