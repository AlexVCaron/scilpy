# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def compute_RCM(array, is_symmetric=True):
    """
    Reverse Cuthill–McKee algorithm permutes a sparse matrix that has a
    symmetric sparsity pattern into a band matrix form with a small bandwidth.

    Parameters
    ----------
    array: ndarray (NxN)
        Sparse connectivity matrix.
    is_symmetric: bool, optional
        Is the matrice symmetric. (if from scil_compute_connectivity.py, True)

    Returns
    -------
    perm: ndarray (N,)
        Output permutations for rows and columns.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')
    csr_arr = csr_matrix(array)
    perm = reverse_cuthill_mckee(csr_arr, symmetric_mode=is_symmetric)

    return perm


def apply_RCM(array, perm):
    """
    Apply the permutation from compute_RCM.

    Parameters
    ----------
    array: ndarray (NxN)
        Sparse connectivity matrix.
    perm: ndarray (N,)
        Permutations for rows and columns to be applied.

    Returns
    -------
    ndarray (N,N)
        Reordered array.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')
    return array[perm].T[perm]


def parse_ordering(in_ordering_file, labels_list=None):
    with open(in_ordering_file, 'r') as my_file:
        lines = my_file.readlines()
        ordering = [[int(val) for val in lines[0].split()],
                    [int(val) for val in lines[1].split()]]
    if labels_list:
        labels_list = np.loadtxt(labels_list,
                                 dtype=np.int16).tolist()
        # If the reordering file refers to labels and not indices
        real_ordering = [[], []]
        real_ordering[0] = [labels_list.index(i) for i in ordering[0]]
        real_ordering[1] = [labels_list.index(i) for i in ordering[1]]
        return real_ordering

    return ordering


def apply_reordering(array, ordering):
    """
    Apply a non-symmetric array ordering that support non-square output.
    The ordering can contain duplicated or discarded rows/columns.

    Parameters
    ----------
    array: ndarray (NxN)
        Sparse connectivity matrix.
    ordering: list of lists
        First elements of the list is the permutation to apply to the rows.
        First elements of the list is the permutation to apply to the columns.

    Returns
    -------
    tmp_array: ndarray (N,N)
        Reordered array.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')
    if not isinstance(ordering, list) or len(ordering) != 2:
        raise ValueError('Ordering should be a list of lists.\n'
                         '[[x1, x2,..., xn], [y1, y2,..., yn]]')
    ind_1, ind_2 = ordering
    if (np.array(ind_1) > array.shape[0]).any() \
            or (ind_2 > np.array(array.shape[1])).any():
        raise ValueError('Indices from configuration are larger than the'
                         'matrix size, maybe you need a labels list?')
    tmp_array = array[tuple(ind_1), :]
    tmp_array = tmp_array[:, tuple(ind_2)]

    return tmp_array
