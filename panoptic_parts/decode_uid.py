"""
Utility functions for reading and writing to our hierarchical panoptic format (see README).
Modified from: https://github.com/pmeletis/panoptic_parts
"""

import functools
from enum import Enum

import numpy as np
import panoptic_parts.utils._sparse_ids_mapping_to_dense_ids_mapping as ndarray_from_dict
import torch

from panoptic_parts.dataset_spec import DatasetSpec


# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [decode_uids, encode_ids]


class Frameworks(Enum):
    PYTHON = "python"
    NUMPY = "numpy"
    TORCH = "torch"


def _validate_uids_values_numpy_python(uids):
    assert isinstance(uids, (int, np.int32, np.ndarray))
    # uids = np.asanyarray(uids, dtype=np.int32)
    if np.any(uids > 99_999_99):
        raise ValueError('Some uids exceed the 99_999_99 encoding limit.')
    if np.any(uids < 0):
        raise ValueError('Some uids are negative.')
    if np.any(np.logical_and(uids >= 100, uids <= 999)):
        raise ValueError(
            'Some uids have length of 3 digits that is not allowed by the encoding format.')


def _decode_uids_functors_and_checking(uids, experimental_noinfo_id):
    # this functions makes the decode_uids more clear by doing all checks here
    # required frameworks: Python and NumPy
    # TODO(panos): split this function into 2
    if not isinstance(experimental_noinfo_id, int):
        raise TypeError('experimental_noinfo_id should be a Python int.')

    if isinstance(uids, np.ndarray):
        if uids.dtype != np.int32:
            raise TypeError(f'{uids.dtype} is an unsupported dtype of np.ndarray uids.')
        where = np.where
        ones_like = np.ones_like
        divmod_ = np.divmod
        maximum = np.maximum
        dtype = np.int32
        logical_and = np.logical_and
        _validate_uids_values_numpy_python(uids)
        return where, ones_like, divmod_, maximum, dtype, logical_and
    if isinstance(uids, (int, np.int32)):
        where = lambda cond, true_br, false_br: true_br if cond else false_br
        ones_like = lambda the_like: type(the_like)(1)
        divmod_ = divmod
        maximum = max
        dtype = (lambda x: x) if isinstance(uids, int) else np.int32
        logical_and = (lambda x, y: x and y) if isinstance(uids, int) else np.logical_and
        _validate_uids_values_numpy_python(uids)
        return where, ones_like, divmod_, maximum, dtype, logical_and

    # use Pytorch framework
    if isinstance(uids, torch.Tensor):
        where = torch.where
        ones_like = torch.ones_like
        divmod_ = lambda x, y: (x // y, x % y)
        maximum = torch.max
        dtype = functools.partial(torch.tensor, dtype=torch.int32)
        logical_and = torch.logical_and
        return where, ones_like, divmod_, maximum, dtype, logical_and

    raise TypeError(f'{type(uids)} is an unsupported type of uids.')


def decode_uids(uids, *, return_sids_iids: bool = False, return_sids_pids: bool = False,
                experimental_noinfo_id: int = -1, experimental_dataset_spec:DatasetSpec=None,
                experimental_correct_range: bool = False):
    """
    Given the universal ids `uids` according to the hierarchical format described
    in README, this function returns element-wise
    the semantic ids (sids), instance ids (iids), and part ids (pids).
    Optionally it returns the sids_iids and sids_pids as well.

    sids_iids represent the semantic-instance-level (two-level) labeling,
    e.g., sids_iids(Cityscapes-Panoptic-Parts) = Cityscapes-Original.

    sids_pids represent the semantic-part-level (semantics) labeling,
    e.g., sids_pids(23_003_04) = 23_04.

    Examples (output is same type as input - not shown for clarity):
      - decode_uids(23, return_sid_pid=True) → (23, -1, -1, 23)
      - decode_uids(23003, return_sid_pid=True) → (23, 3, -1, 23)
      - decode_uids(2300304, return_sid_pid=True) → (23, 3, 4, 2304)
      - decode_uids(tf.constant([1, 12, 1234, 12345, 123456, 1234567])) →
        ([ 1, 12,   1,  12,   1,  12],
         [-1, -1, 234, 345, 234, 345],
         [-1, -1,  -1,  -1,  56,  67])
      - decode_uids(np.array([[1, 12], [1234, 12345]])) →
        ([[ 1, 12], [ 1, 12]],
         [[ -1,  -1], [234, 345]],
         [[-1, -1], [-1, -1]])

    Args:
      uids: tf.Tensor of dtype tf.int32 and arbitrary shape,
            or np.ndarray of dtype np.int32 and arbitrary shape,
            or torch.tensor of dtype torch.int32 and arbitrary shape,
            or Python int,
            or np.int32 integer,
            with elements according to hierarchical format (see README).

    Returns:
      sids, iids, pids: same type and shape as uids, with -1 for not relevant pixels.

      sids will have no -1.
      iids will have -1 for pixels labeled with sids only.
      pids will have -1 for pixels labeled with sids or sids_iids only.

      if return_sids_iids:
        sids_iids: same type and shape as uids, will have no -1.
      if return_sids_pids:
        sids_pids: same type and shape as uids, will have no -1.
    """
    where, ones_like, divmod_, maximum, dtype, logical_and = _decode_uids_functors_and_checking(
        uids, experimental_noinfo_id)

    # this function uses dtype and np.asanyarray because:
    # dtype: numpy implicitly converts Python int literals in np.int64, we need np.int32
    # np.asanyarray: numpy implicitly converts ndarray with one element to np.int32 (which is not
    #   ndarray), moreover dtypes are implicitly converted to np.int64 for arithmetic operations

    # pad uids to uniform full format for easier handling: uids = 0 or uids in [1_000_00, 99_999_99],
    # this generates a fake iid (0) for uids <= 99, and noinfo pid (0) for uids <= 99_999, but these
    # case are handled later
    uids_padded = where(uids <= 99_999,
                        where(uids <= 99, uids * dtype(10 ** 5), uids * dtype(10 ** 2)),
                        uids)
    # split uids to components (sids, iids, pids) from right to left
    sids_iids, pids = divmod_(uids_padded, dtype(10 ** 2))
    sids, iids = divmod_(sids_iids, dtype(10 ** 3))
    # handle fake iids, pids introduced before for sids_iids, iids, pids
    sids_iids = where(uids <= 99_999, uids, sids_iids)
    noinfo_ids = ones_like(uids) * dtype(experimental_noinfo_id)
    iids = where(uids <= 99, noinfo_ids, iids)
    pids = where(logical_and(uids <= 99_999, pids == dtype(0)), noinfo_ids, pids)  # pid=0 has no info

    # use experimental_dataset_spec to re-set any values that may reside outside of
    # valid ranges for sids, iids, sids_iids, this is used for sanity/validity
    # sids are set to 0 (unlabeled) if not in correct range
    # iids are set to -1 (no info) if sids not in the correct range
    # pids and sids_pids are re-set later after removing the part-level instance information layer
    if (experimental_correct_range == True and
            experimental_dataset_spec is not None):
        if not isinstance(uids, (np.ndarray, np.int32)):
            raise NotImplementedError(
                f'range correction is only supported for np.ndarray and np.int32 for now.')
        sids_allowed = list(experimental_dataset_spec.sid2scene_class)
        sids_ok = np.isin(sids, sids_allowed)
        # if not np.all(sids_ok):
        #   print(f'Found {np.count_nonzero(np.invert(sids_ok))} pixels with invalid sids. The (old, new) tuples are:',
        #         compare_pixelwise(sids, np.where(sids_ok, sids, dtype(0))), sep='\n')
        sids = np.where(sids_ok, sids, dtype(0))
        iids = np.where(sids_ok, iids, noinfo_ids)
        sids_iids = np.where(sids_ok, sids_iids, dtype(0))

    # A mapping of pids in order to remove the part-level instance information layer,
    # as Part-aware Panoptic Segmentation does not require this layer,
    # this only applies to PPP for now.
    if (experimental_dataset_spec is not None and
            hasattr(experimental_dataset_spec, '_sid_pid_file2sid_pid')):
        if not isinstance(uids, (np.ndarray, np.int32)):
            raise NotImplementedError(
                f'sid_pid from file mapping is only supported for np.ndarray and np.int32 for now. '
                f'Given uids of type: {type(uids)}.')
        spf2sp = experimental_dataset_spec._sid_pid_file2sid_pid
        if 'sids_pids' not in locals():  # save some compute
            sids_pids = where(pids == noinfo_ids, sids, sids * dtype(10 ** 2) + pids)
        spf2sp__dense = ndarray_from_dict(spf2sp, -10 ** 6, length=10000)  # -10**6 random number
        sids_pids = spf2sp__dense[sids_pids]
        assert not np.any(np.equal(sids_pids, -10 ** 6)), (
            'Unhandled case: experimental_dataset_spec._sid_pid_file2sid_pid does not '
            'contain all pids in GT files. Raise an issue to maintainers.')
        # TODO(panos): only pids are mapped from now (this follows from the allowed mapping format in yaml)
        pids = where(sids_pids >= 1_00, sids_pids % 100, noinfo_ids)

    # use experimental_dataset_spec to re-set any values that may reside outside of
    # valid ranges for pids, sids_pids, this is used for sanity/validity
    # pids are set to -1 (no info) if pids of sids not in correct range
    # sids_pids are set to 0 (unlabeled) if sids not in correct range or corrected sids
    #   if pids not in the correct range
    if (experimental_correct_range == True and
            experimental_dataset_spec is not None):
        sids_pids_allowed = list(experimental_dataset_spec.sid_pid2scene_class_part_class)
        sids_pids = where(np.logical_or(sids == dtype(0), pids == noinfo_ids),
                          sids,
                          sids * dtype(10 ** 2) + pids)
        sids_pids_ok = np.isin(sids_pids, sids_pids_allowed)
        # if not np.all(sids_pids_ok):
        #   print(f'Found {np.count_nonzero(np.invert(sids_pids_ok))} pixels with invalid sids. The (old, new) tuples are:',
        #         compare_pixelwise(sids_pids, np.where(sids_pids_ok, sids_pids, sids)), sep='\n')
        sids_pids = np.where(sids_pids_ok, sids_pids, sids)
        pids = where(sids_pids >= 1_00, sids_pids % 100, noinfo_ids)

    if isinstance(uids, np.ndarray):
        sids = np.asanyarray(sids, dtype=np.int32)
    returns = (sids, iids, pids)

    if return_sids_iids:
        returns += (sids_iids,)

    if return_sids_pids:
        if 'sids_pids' not in locals():  # save some compute
            sids_pids = where(pids == noinfo_ids, sids, sids * dtype(10 ** 2) + pids)
        if isinstance(uids, np.ndarray):
            sids_pids = np.asanyarray(sids_pids, dtype=np.int32)
        returns += (sids_pids,)

    return returns


def _validate_ids_values_numpy_python(sids, iids, pids):
    assert isinstance(sids, (int, np.int32, np.ndarray))
    assert type(sids) is type(iids) and type(iids) is type(pids)
    sids, iids, pids = map(functools.partial(np.array, dtype=np.int32), [sids, iids, pids])
    if np.any(sids > 99):
        raise ValueError('Some sids exceed the 99 encoding limit.')
    if np.any(iids > 999):
        raise ValueError('Some iids exceed the 999 encoding limit.')
    if np.any(pids > 99):
        raise ValueError('Some sids exceed the 99 encoding limit.')
    if np.any(sids < -1):
        raise ValueError('Some sids are negative.')
    if np.any(iids < -1):
        raise ValueError('Some iids are negative.')
    if np.any(pids < -1):
        raise ValueError('Some pids are negative.')


def _encode_ids_functors_and_checking(sids, iids, pids):
    # this functions makes the endoce_ids more clear
    # required frameworks: Python and NumPy
    if type(sids) is not type(iids) and type(iids) is not type(pids):
        raise ValueError(
            f"All arguments must have the same type, given {(*map(type, (sids, iids, pids)),)}")
    if isinstance(sids, np.ndarray):
        if sids.dtype != np.int32:
            raise TypeError(f'{sids.dtype} is an unsupported dtype of np.ndarray ids.')
        where = np.where
        _validate_ids_values_numpy_python(sids, iids, pids)
        return where
    if isinstance(sids, (int, np.int32)):
        where = lambda cond, true_br, false_br: true_br if cond else false_br
        _validate_ids_values_numpy_python(sids, iids, pids)
        return where

    # Use Pytorch
    if isinstance(sids, torch.Tensor):
        where = torch.where
        return where

    raise TypeError(f'{type(sids)} is an unsupported type of ids.')


def encode_ids(sids, iids, pids):
    """
    Given semantic ids (sids), instance ids (iids), and part ids (pids)
    this function encodes them element-wise to uids
    according to the hierarchical format described in README.

    This function is the opposite of decode_uids, i.e.,
    uids = encode_ids(decode_uids(uids)).

    Args:
      sids, iids, pids: all of the same type with -1 for non-relevant pixels with
        elements according to hierarchical format (see README). Can be:
        tf.Tensor of dtype tf.int32 and arbitrary shape,
        or np.ndarray of dtype np.int32 and arbitrary shape,
        or torch.tensor of dtype torch.int32 and arbitrary shape,
        or Python int,
        or np.int32 integer.

    Return:
      uids: same type and shape as the args according to hierarchical format (see README).
    """
    # TODO!!!(panos): encode_ids(sid, -1, pid) ends with a sid_pid and not a uid,
    #   is this behavior desirable?
    where = _encode_ids_functors_and_checking(sids, iids, pids)

    sids_iids = where(iids < 0, sids, sids * 10 ** 3 + iids)
    uids = where(pids < 0, sids_iids, sids_iids * 10 ** 2 + pids)

    return uids