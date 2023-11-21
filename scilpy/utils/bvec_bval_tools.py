# -*- coding: utf-8 -*-

import logging
from enum import Enum

import numpy as np

from scilpy.image.utils import volume_iterator
from scilpy.gradientsampling.load_gradient_sampling import (
    load_gradient_sampling_fsl,
    load_gradient_sampling_mrtrix,
    load_gradient_sampling_siemens)
from scilpy.gradientsampling.save_gradient_sampling import (
    save_gradient_sampling_fsl,
    save_gradient_sampling_mrtrix,
    save_gradient_sampling_siemens)


DEFAULT_B0_THRESHOLD = 20
SUPPORTED_FORMATS = ["fsl", "mrtrix", "siemens"]
SIEMENS_NORMALIZATION = ["none", "unity", "maximum"]

class B0ExtractionStrategy(Enum):
    FIRST = "first"
    MEAN = "mean"
    ALL = "all"


class GradientProtocol:
    def __init__(self, bvals=None, shell_idx=None, points=None):
        self.points = points
        self.bvals = bvals
        self.shell_idx = shell_idx
        self.is_initialized = not any(
            [s is None for s in [points, bvals, shell_idx]])

    def _normalize(self, _bv, _nb, _nrm):
        if _nrm == "none":
            return np.sqrt(_bv / _nb)
        elif _nrm == "unity":
            return np.ones_like(_bv)
        elif _nrm == "maximum":
            return _bv / _nb

    @classmethod
    def from_fsl(cls, bval, bvec):
        return GradientProtocol(*load_gradient_sampling_fsl(bval, bvec))

    @classmethod
    def from_mrtrix(cls, protocol_b):
        return GradientProtocol(*load_gradient_sampling_mrtrix(protocol_b))

    @classmethod
    def from_siemens(cls, protocol_dvs, b_nominal=1., ref_affine=np.eye(4)):
        return GradientProtocol(*load_gradient_sampling_siemens(
            protocol_dvs, b_nominal, ref_affine))

    def save(
        self, target_format, output_fname,
        b_nominal=1., normalization="none", ref_affine=np.eye(4)
    ):
        if not self.is_initialized:
            logging.error(
                "Gradient protocol writer has no data to write. Either supply "
                "the directions, bvalues, shells when creating the writer, or "
                "initialize using one of the reading factories :\n"
                "  - GradientProtocol.from_fsl(bval, bvec)\n"
                "  - GradientProtocol.from_mrtrix(b)\n"
                "  - GradientProtocol.from_siemens(dvs)")

        if target_format == "fsl":
            bval = "{}.bval".format(output_fname)
            bvec = "{}.bvec".format(output_fname)
            save_gradient_sampling_fsl(self.points,
                                       self.shell_idx,
                                       self.bvals,
                                       filename_bval=bval,
                                       filename_bvec=bvec)
        elif target_format == "mrtrix":
            save_gradient_sampling_mrtrix(self.points,
                                          self.shell_idx,
                                          self.bvals,
                                          "{}.b".format(output_fname))
        elif target_format == "siemens":
            if len(self.bvals) > 1 and normalization == "unity":
                logging.error('ERROR: unity normalization is valid only ' + 
                              'if all directions have the same b-value')

            b_nominal = b_nominal or max(self.bvals)
            rot_to_image = np.eye(4) if ref_affine is None else ref_affine
            save_gradient_sampling_siemens(
                np.linalg.inv(rot_to_image) @ self.points,
                self.shell_idx,
                self._normalize(self.bvals, b_nominal, normalization),
                normalization,
                "{}.dvs".format(output_fname))


def mrtrix2fsl(mrtrix_file, fsl_bval, fsl_bvec):
    bvals, shell_idx, points = load_gradient_sampling_mrtrix(mrtrix_file)
    save_gradient_sampling_fsl(points, shell_idx, bvals, fsl_bval, fsl_bvec)


def fsl2mrtrix(fsl_bval, fsl_bvec, mrtrix_file):
    bvals, shell_idx, points = load_gradient_sampling_fsl(fsl_bval, fsl_bvec)
    save_gradient_sampling_mrtrix(points, shell_idx, bvals, mrtrix_file)


def fsl2siemens(
    fsl_bval, fsl_bvec, siemens_dvs,
    ref_affine=None, b_nominal=None, normalization="none"
):
    def _normalize(_bv, _nb, _nrm):
        if _nrm == "none":
            return np.sqrt(_bv / _nb)
        elif _nrm == "unity":
            return np.ones_like(_bv)
        elif _nrm == "maximum":
            return _bv / _nb

    bvals, shell_idx, points = load_gradient_sampling_fsl(fsl_bval, fsl_bvec)

    if len(bvals) > 1 and normalization == "unity":
        logging.error('ERROR: unity normalization is valid only ' + 
                        'if all directions have the same b-value')

    b_nominal = b_nominal or max(bvals)
    rot_to_image = np.eye(4) if ref_affine is None else ref_affine
    save_gradient_sampling_siemens(
        np.linalg.inv(rot_to_image) @ points,
        shell_idx,
        _normalize(bvals, b_nominal, normalization),
        normalization,
        siemens_dvs)


def is_normalized_bvecs(bvecs):
    """
    Check if b-vectors are normalized.

    Parameters
    ----------
    bvecs : (N, 3) array
        input b-vectors (N, 3) array

    Returns
    -------
    True/False
    """

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3,
                                bvecs_norm == 0))


def normalize_bvecs(bvecs, filename=None):
    """
    Normalize b-vectors

    Parameters
    ----------
    bvecs : (N, 3) array
        input b-vectors (N, 3) array
    filename : string
        output filename where to save the normalized bvecs

    Returns
    -------
    bvecs : (N, 3)
       normalized b-vectors
    """

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    idx = bvecs_norm != 0
    bvecs[idx] /= bvecs_norm[idx, None]

    if filename is not None:
        logging.info('Saving new bvecs: {}'.format(filename))
        np.savetxt(filename, np.transpose(bvecs), "%.8f")

    return bvecs


def check_b0_threshold(
    force_b0_threshold, bvals_min, b0_thr=DEFAULT_B0_THRESHOLD
):
    """Check if the minimal bvalue is under zero or over the threshold.
    If `force_b0_threshold` is true, don't raise an error even if the minimum
    bvalue is over the threshold.

    Parameters
    ----------
    force_b0_threshold : bool
        If True, don't raise an error.
    bvals_min : float
        Minimum bvalue.
    b0_thr : float
        Maximum bvalue considered as a b0.

    Raises
    ------
    ValueError
        If the minimal bvalue is over the threshold, and
        `force_b0_threshold` is False.
    """
    if b0_thr > DEFAULT_B0_THRESHOLD:
        logging.warning(
            'Warning: Your defined threshold is {}. This is suspicious. We '
            'recommend using volumes with bvalues no higher '
            'than {} as b0s.'.format(b0_thr, DEFAULT_B0_THRESHOLD)
        )

    if bvals_min < 0:
        logging.warning(
            'Warning: Your dataset contains negative b-values (minimal '
            'bvalue of {}). This is suspicious. We recommend you check '
            'your data.')

    if bvals_min > b0_thr:
        if force_b0_threshold:
            logging.warning(
                'Warning: Your minimal bvalue is {}, but the threshold '
                'is set to {}. Since --force_b0_threshold was specified, '
                'the script will proceed with a threshold of {}'
                '.'.format(bvals_min, b0_thr, bvals_min))
            return bvals_min
        else:
            raise ValueError('The minimal bvalue ({}) is greater than the '
                             'threshold ({}). No b0 volumes can be found.\n'
                             'Please check your data to ensure everything '
                             'is correct.\n'
                             'Use --force_b0_threshold to execute '
                             'regardless.'
                             .format(bvals_min, b0_thr))

    return b0_thr


def get_shell_indices(bvals, shell, tol=10):
    """
    Get shell indices

    Parameters
    ----------
    bvals: array (N,)
        array of bvals
    shell: list
        list of bvals
    tol: int
        tolerance to accept a bval

    Returns
    -------
        numpy.ndarray where shells are found
    """

    return np.where(
        np.logical_and(bvals < shell + tol, bvals > shell - tol))[0]


def identify_shells(bvals, threshold=40.0, roundCentroids=False, sort=False):
    """
    Guessing the shells from the b-values. Returns the list of shells and, for
    each b-value, the associated shell.

    Starting from the first shell as holding the first b-value in bvals,
    the next b-value is considered on the same shell if it is closer than
    threshold, or else we consider that it is on another shell. This is an
    alternative to K-means considering we don't already know the number of
    shells K.

    Note. This function should be added in Dipy soon.

    Parameters
    ----------
    bvals: array (N,)
        Array of bvals
    threshold: float
        Limit value to consider that a b-value is on an existing shell. Above
        this limit, the b-value is placed on a new shell.
    roundCentroids: bool
        If true will round shell values to the nearest 10.
    sort: bool
        Sort centroids and shell_indices associated.

    Returns
    -------
    centroids: array (K)
        Array of centroids. Each centroid is a b-value representing the shell.
        K is the number of identified shells.
    shell_indices: array (N,)
        For each bval, the associated centroid K.
    """
    if len(bvals) == 0:
        raise ValueError('Empty b-values.')

    # Finding centroids
    bval_centroids = [bvals[0]]
    for bval in bvals[1:]:
        diffs = np.abs(np.asarray(bval_centroids, dtype=float) - bval)
        if not len(np.where(diffs < threshold)[0]):
            # Found no bval in bval centroids close enough to the current one.
            # Create new centroid (i.e. new shell)
            bval_centroids.append(bval)
    centroids = np.array(bval_centroids)

    # Identifying shells
    bvals_for_diffs = np.tile(bvals.reshape(bvals.shape[0], 1),
                              (1, centroids.shape[0]))

    shell_indices = np.argmin(np.abs(bvals_for_diffs - centroids), axis=1)

    if roundCentroids:
        centroids = np.round(centroids, decimals=-1)

    if sort:
        sort_index = np.argsort(centroids)
        sorted_centroids = np.zeros(centroids.shape)
        sorted_indices = np.zeros(shell_indices.shape)
        for i in range(len(centroids)):
            sorted_centroids[i] = centroids[sort_index[i]]
            sorted_indices[shell_indices == i] = sort_index[i]
        return sorted_centroids, sorted_indices

    return centroids, shell_indices


def extract_dwi_shell(dwi, bvals, bvecs, bvals_to_extract, tol=20,
                      block_size=None):
    """Extracts the DWI volumes that are on specific b-value shells. Many
    shells can be extracted at once by specifying multiple b-values. The
    extracted volumes are in the same order as in the original file.

    If the b-values of a shell are not all identical, use the --tolerance
    argument to adjust the accepted interval. For example, a b-value of 2000
    and a tolerance of 20 will extract all volumes with a b-values from 1980 to
    2020.

    Files that are too large to be loaded in memory can still be processed by
    setting the --block-size argument. A block size of X means that X DWI
    volumes are loaded at a time for processing.

    Parameters
    ----------
    dwi : nib.Nifti1Image
        Original multi-shell volume.
    bvals : ndarray
        The b-values in FSL format.
    bvecs : ndarray
        The b-vectors in FSL format.
    bvals_to_extract : list of int
        The list of b-values to extract.
    tol : int
        The tolerated gap between the b-values to extract and the actual
        b-values.
    block_size : int
        Load the data using this block size. Useful when the data is too
        large to be loaded in memory.

    Returns
    -------
    indices : ndarray
        Indices of the volumes corresponding to the provided b-values.
    shell_data : ndarray
        Volumes corresponding to the provided b-values.
    output_bvals : ndarray
        Selected b-values.
    output_bvecs : ndarray
        Selected b-vectors.

    """
    indices = [get_shell_indices(bvals, shell, tol=tol)
               for shell in bvals_to_extract]
    indices = np.unique(np.sort(np.hstack(indices)))

    if len(indices) == 0:
        raise ValueError("There are no volumes that have the supplied b-values"
                         ": {}".format(bvals_to_extract))

    logging.info(
        "Extracting shells [{}], with number of images per shell [{}], "
        "from {} images from {}."
        .format(" ".join([str(b) for b in bvals_to_extract]),
                " ".join([str(len(get_shell_indices(bvals, shell, tol=tol)))
                          for shell in bvals_to_extract]),
                len(bvals), dwi.get_filename()))

    if block_size is None:
        block_size = dwi.shape[-1]

    # Load the shells by iterating through blocks of volumes. This approach
    # is slower for small files, but allows very big files to be split
    # with less memory usage.
    shell_data = np.zeros((dwi.shape[:-1] + (len(indices),)))
    for vi, data in volume_iterator(dwi, block_size):
        in_volume = np.array([i in vi for i in indices])
        in_data = np.array([i in indices for i in vi])
        shell_data[..., in_volume] = data[..., in_data]

    output_bvals = bvals[indices].astype(int)
    output_bvals.shape = (1, len(output_bvals))
    output_bvecs = bvecs[indices, :]

    return indices, shell_data, output_bvals, output_bvecs


def extract_b0(dwi, b0_mask, extract_in_cluster=False,
               strategy=B0ExtractionStrategy.MEAN, block_size=None):
    """
    Extract a set of b0 volumes from a dwi dataset

    Parameters
    ----------
    dwi : nib.Nifti1Image
        Original multi-shell volume.
    b0_mask: array of bool
        Mask over the time dimension (4th) identifying b0 volumes.
    extract_in_cluster: bool
        Specify to extract b0's in each continuous sets of b0 volumes
        appearing in the input data.
    strategy: Enum
        The extraction strategy, of either select the first b0 found, select
        them all or average them. When used in conjunction with the batch
        parameter set to True, the strategy is applied individually on each
        continuous set found.
    block_size : int
        Load the data using this block size. Useful when the data is too
        large to be loaded in memory.

    Returns
    -------
    b0_data : ndarray
        Extracted b0 volumes.
    """

    indices = np.where(b0_mask)[0]

    if block_size is None:
        block_size = dwi.shape[-1]

    if not extract_in_cluster and strategy == B0ExtractionStrategy.FIRST:
        idx = np.min(indices)
        output_b0 = dwi.dataobj[..., idx:idx + 1].squeeze()
    else:
        # Generate list of clustered b0 in the data
        mask = np.ma.masked_array(b0_mask)
        mask[~b0_mask] = np.ma.masked
        b0_clusters = np.ma.notmasked_contiguous(mask, axis=0)

        if extract_in_cluster or strategy == B0ExtractionStrategy.ALL:
            if strategy == B0ExtractionStrategy.ALL:
                time_d = len(indices)
            else:
                time_d = len(b0_clusters)

            output_b0 = np.zeros(dwi.shape[:-1] + (time_d,))

            for idx, cluster in enumerate(b0_clusters):
                if strategy == B0ExtractionStrategy.FIRST:
                    data = dwi.dataobj[..., cluster.start:cluster.start + 1]
                    output_b0[..., idx] = data.squeeze()
                else:
                    vol_it = volume_iterator(dwi, block_size,
                                             cluster.start, cluster.stop)

                    for vi, data in vol_it:
                        if strategy == B0ExtractionStrategy.ALL:
                            in_volume = np.array([i in vi for i in indices])
                            output_b0[..., in_volume] = data
                        elif strategy == B0ExtractionStrategy.MEAN:
                            output_b0[..., idx] += np.sum(data, -1)

                    if strategy == B0ExtractionStrategy.MEAN:
                        output_b0[..., idx] /= cluster.stop - cluster.start

        else:
            output_b0 = np.zeros(dwi.shape[:-1])
            for cluster in b0_clusters:
                vol_it = volume_iterator(dwi, block_size,
                                         cluster.start, cluster.stop)

                for _, data in vol_it:
                    output_b0 += np.sum(data, -1)

            output_b0 /= len(indices)

    return output_b0


def flip_mrtrix_gradient_sampling(gradient_sampling_filename,
                                  gradient_sampling_flipped_filename, axes):
    """
    Flip Mrtrix gradient sampling on a axis

    Parameters
    ----------
    gradient_sampling_filename: str
        Gradient sampling filename
    gradient_sampling_flipped_filename: str
        Gradient sampling flipped filename
    axes: list of int
        List of axes to flip (e.g. [0, 1])
    """
    gradient_sampling = np.loadtxt(gradient_sampling_filename)
    for axis in axes:
        gradient_sampling[:, axis] *= -1

    np.savetxt(gradient_sampling_flipped_filename,
               gradient_sampling,
               "%.8f %.8f %.8f %0.6f")


def flip_fsl_gradient_sampling(bvecs_filename, bvecs_flipped_filename, axes):
    """
    Flip FSL bvecs on a axis

    Parameters
    ----------
    bvecs_filename: str
        Bvecs filename
    bvecs_flipped_filename: str
        Bvecs flipped filename
    axes: list of int
        List of axes to flip (e.g. [0, 1])
    """
    bvecs = np.loadtxt(bvecs_filename)
    for axis in axes:
        bvecs[axis, :] *= -1

    np.savetxt(bvecs_flipped_filename, bvecs, "%.8f")


def swap_fsl_gradient_axis(bvecs_filename, bvecs_swapped_filename, axes):
    """
    Swap FSL bvecs

    Parameters
    ----------
    bvecs_filename: str
        Bvecs filename
    bvecs_swapped_filename: str
        Bvecs swapped filename
    axes: list of int
        List of axes to swap (e.g. [0, 1])
    """

    bvecs = np.loadtxt(bvecs_filename)
    new_bvecs = np.copy(bvecs)
    new_bvecs[axes[0], :] = bvecs[axes[1], :]
    new_bvecs[axes[1], :] = bvecs[axes[0], :]

    np.savetxt(bvecs_swapped_filename, new_bvecs, "%.8f")


def swap_mrtrix_gradient_axis(bvecs_filename, bvecs_swapped_filename, axes):
    """
    Swap MRtrix bvecs

    Parameters
    ----------
    bvecs_filename: str
        Bvecs filename
    bvecs_swapped_filename: str
        Bvecs swapped filename
    axes: list of int
        List of axes to swap (e.g. [0, 1])
    """

    bvecs = np.loadtxt(bvecs_filename)
    new_bvecs = np.copy(bvecs)

    new_bvecs[:, axes[0]] = bvecs[:, axes[1]]
    new_bvecs[:, axes[1]] = bvecs[:, axes[0]]

    np.savetxt(bvecs_swapped_filename,
               new_bvecs,
               "%.8f %.8f %.8f %0.6f")
