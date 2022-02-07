import copy
import glob
import os
import os.path as op
from typing import Dict, Union

import numpy as np
from PIL import Image

from panoptic_parts.decode_uid import *


# cyclic import
# from panoptic_parts.utils.format import decode_uids, encode_ids

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [safe_write]


def _sparse_ids_mapping_to_dense_ids_mapping(ids_dict, void, length=None, dtype=np.int32):
    """
    Create a dense np.array from an ids dictionary. The array can be used
    for indexing, e.g. numpy advanced indexing or tensorflow gather.
    This method is useful to transform a dictionary of uids to class mappings (e.g. {2600305: 3}),
    to a dense np.array that has in position 2600305 the value 3. This in turn can be used in
    gathering operations. The reason that the mapping is given in a dictionary is due to its
    sparseness, e.g. we may not want to hard-code an array with 2600305 elements in order to
    have the mapping for the 2600305th element.

    ids.values() and void must have the same shape and dtype.

    The length of the dense_mapping is infered from the maximum value of ids_dict.keys(). If you
    need a longer dense_mapping provide the length in `length`.

    Args:
      ids_dict: dictionary mapping ids to numbers (usually classes),
      void: int, list of int, tuple of int, the positions of the dense array that don't appear in
        ids_dict.keys() will be filled with the void value,
      length: the length of the dense mapping can be explicitly provided
      dtype: the dtype of the returned dense mapping
    """
    # TODO(panos): add args requirements checking, and refactor this code
    # TODO(panos): check the validity of +1 (useful only if key 0 exists?)
    allowed_np_int_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
    if not isinstance(ids_dict, dict):
        raise TypeError('ids_dict must be a dictionary.')
    if not (isinstance(void, int) or (
            isinstance(void, (list, tuple)) and all(map(lambda x: isinstance(x, int), void)))):
        raise TypeError('void must be an int type integer or a list of int type integers.')
    if length is not None and not isinstance(length, int):
        raise TypeError(f'length must be None or an int type integer, but was {length}.')
    if not any(map(lambda x: dtype is x, allowed_np_int_types)):
        raise TypeError(f'dtype must be one of the following integer types: {allowed_np_int_types}, but was {dtype}.')

    void_np = np.array(void)
    length_mapping = length or np.max(list(ids_dict.keys())) + 1

    if void_np.ndim == 0:
        dense_mapping = np.full(length_mapping, void, dtype=dtype)
        for uid, cid in ids_dict.items():
            dense_mapping[uid] = cid
    elif void_np.ndim == 1:
        dense_mapping = np.full((length_mapping, void_np.shape[0]), void, dtype=dtype)
        for uid, cid in ids_dict.items():
            dense_mapping[uid] = cid
    else:
        raise NotImplementedError('Not yet implemented.')

    return dense_mapping


def safe_write(path, image):
    """
    Check if `path` exist and if it doesn't creates all needed intermediate-level directories
    and saves `image` to `path`.

    Args:
      path: a path passed to os.path.exists, os.makedirs and PIL.Image.save()
      image: a numpy image passed to PIL.Image.fromarray()

    Return:
      False is path exists. True if the `image` is successfully written.
    """
    if op.exists(path):
        print('File already exists:', path)
        return False

    os.makedirs(op.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path)
    return True


def uids_lids2uids_cids(uids_with_lids, lids2cids):
    """
    Convert uids with semantic classes encoded as lids to uids with cids.
    This function is useful in the Cityscapes context, or other datasets
    where the lids and cids separation is used.
    """
    uids = uids_with_lids
    sids, _, _ = decode_uids(uids)
    uids_with_cids = np.where(
        uids <= 99,
        lids2cids[sids],
        np.where(uids <= 99_999,
                 lids2cids[sids] * 10 ** 3 + uids % 10 ** 3,
                 lids2cids[sids] * 10 ** 5 + uids % 10 ** 5))

    return uids_with_cids


def color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def _parse_sid_group(sid, sid_group):
    assert sid >= 1
    # import itertools
    # pids_all = itertools.chain.from_iterable(sid_group)
    sid_pid_old2sid_pid_new = dict()
    for i, pids_group in enumerate(sid_group, start=1):
        for pid in pids_group:
            sid_pid_old2sid_pid_new[sid * 100 + pid] = sid * 100 + i
    return sid_pid_old2sid_pid_new


def _transform_uids(uids, max_sid, sid2pids_groups):
    sid_pid_old2sid_pid_new = dict()
    for sid, sid_group in sid2pids_groups.items():
        sid_pid_old2sid_pid_new.update(_parse_sid_group(sid, sid_group))
    sid_pid_old2sid_pid_new[0] = 0
    for sid in range(1, max_sid + 1):
        if sid not in sid_pid_old2sid_pid_new.keys():
            sid_pid_old2sid_pid_new[sid] = sid
        for pid in range(100):
            sid_pid = sid * 100 + pid
            if pid == 0:
                sid_pid_old2sid_pid_new[sid_pid] = sid
                continue
            if sid_pid not in sid_pid_old2sid_pid_new.keys():
                sid_pid_old2sid_pid_new[sid_pid] = sid
    sid_pid_old2sid_pid_new = dict(sorted(sid_pid_old2sid_pid_new.items()))
    palette = np.asarray(list(sid_pid_old2sid_pid_new.values()), dtype=np.int32)

    _, iids, _, sids_pids = decode_uids(uids, return_sids_pids=True)
    sids_pids = palette[sids_pids]
    sids = np.where(sids_pids <= 99, sids_pids, sids_pids // 100)
    pids = np.where(sids_pids <= 99, -1, sids_pids % 100)
    return encode_ids(sids, iids, pids)


def _print_metrics_from_confusion_matrix(cm,
                                         names=None,
                                         printfile=None,
                                         printcmd=False,
                                         summary=False,
                                         ignore_ids=list()):
    # cm: numpy, 2D, square, np.int32 or np.int64 array, not containing NaNs
    # names: python list of names
    # printfile: file handler or None to print metrics to file
    # printcmd: True to print a summary of metrics to terminal
    # summary: if printfile is not None, prints only a summary of metrics to file
    # ignore_ids: ids in cm to ignore, zero-based
    #  this is needed for example in the case of UNLABELED pixels:
    #  it will be like pixels with those classes do not exist in the dataset and
    #  predictions in these classes are not made

    # Common situations in the CM:
    # i) there is an "UNLABELED" class in the dataset:
    #   since "UNLABELED" pixels is like they do not exist in the dataset we want them
    #   to be ignored from the metrics, however if the network predicts the "UNLABELED"
    #   class, then some pixels belonging to other classes can be predicted as "UNLABELED"
    # ii) there are some classes that have no pixels in the evaluation dataset,
    #   this results in the row of those classes to have all zeros (tp_fp = 0),
    #   we can safely remove these rows, however we cannot remove the column of these classes
    #   because they act as FP to the other classes

    # Solutions:
    # i) this case involves definition of the user of what is "UNLABELED" so it must be
    #   provided in the ignore_ids
    # ii) this case is automatically found (all zero rows) and can be handled
    # For both cases we remove the rows from the CM, and we aggregate the columns into
    #   a new IGNORE column in order to keep the FP for all other classes, this leads
    #   to a non-square CM, having more columns than rows

    # sanity checks
    assert isinstance(cm, np.ndarray), 'Confusion matrix must be numpy array.'
    cms = cm.shape
    assert all([cm.dtype in [np.int32, np.int64],
                cm.ndim == 2,
                cms[0] == cms[1],
                not np.any(np.isnan(cm))]), (
        f"Check print_metrics_from_confusion_matrix input requirements. "
        f"Input has {cm.ndim} dims, is {cm.dtype}, has shape {cms[0]}x{cms[1]} "
        f"or may contain NaNs.")
    if not names:
        names = ['unknown'] * cms[0]
    assert len(names) == cms[0], (
        f"names ({len(names)}) must be enough for indexing confusion matrix ({cms[0]}x{cms[1]}).")
    # assert os.path.isfile(printfile), 'printfile is not a file.'
    if ignore_ids:
        # convention: ids start from 0
        assert all([min(ignore_ids) >= 0, max(ignore_ids) <= cms[0] - 1]), (
            f"Ignore ids {np.unique(ignore_ids)} not in correct range [0, {cms[0] - 1}].")

    # refine CM
    # accumulate all ignored classes FP into a new column
    # remove ignored rows and columns from the CM
    extra_fp = np.zeros_like(cm[0])
    # add an assertion for next np.sum(cm,1) > 0
    ids_class_not_exist = np.nonzero(np.equal(np.sum(cm, 1), 0))[0]
    ids_remove = set(ids_class_not_exist) | set(ignore_ids)
    for id_remove in ids_remove:
        extra_fp += cm[:, id_remove]
    ids_keep = list(set(range(cms[0])) - ids_remove)
    cm = cm[:, ids_keep][ids_keep, :]
    extra_fp = extra_fp[ids_keep]
    names_new = list(map(lambda t: t[1], filter(lambda t: t[0] in ids_keep, enumerate(names))))

    # metric computations
    tp = np.diagonal(cm)
    tp_fp = np.sum(cm, 1) + extra_fp
    tp_fn = np.sum(cm, 0)
    accuracies = tp / tp_fp * 100
    ious = tp / (tp_fn + tp_fp - tp) * 100
    # summarize per-class metrics
    global_accuracy = np.trace(cm) / np.sum(cm) * 100
    mean_accuracy = np.mean(accuracies)
    mean_iou = np.mean(ious)

    # reporting
    names_ignored = list(map(lambda t: t[1], filter(lambda t: t[0] in ids_remove, enumerate(names))))
    log_string = "\n"
    log_string += f"Ignored classes ({len(ids_remove)}/{cms[0]}): {names_ignored}.\n"
    log_string += "Per class accuracies and ious:\n"
    for l, a, i in zip(names_new, accuracies, ious):
        log_string += f"{l:<30s}  {a:>5.2f}  {i:>5.2f}\n"
    num_classes_average = len(ids_keep)
    log_string += f"Global accuracy: {global_accuracy:5.2f}\n"
    log_string += f"Average accuracy ({num_classes_average}): {mean_accuracy:5.2f}\n"
    log_string += f"Average iou ({num_classes_average}): {mean_iou:5.2f}\n"

    if printcmd:
        print(log_string)

    if printfile:
        if summary:
            printfile.write(log_string)
        else:
            print(f"{global_accuracy:>5.2f}",
                  f"{mean_accuracy:>5.2f}",
                  f"{mean_iou:>5.2f}",
                  accuracies,
                  ious,
                  file=printfile)


def compare_pixelwise(l1, l2):
    """
    Compare numpy arrays l1, l2 with same shape and dtype in a pixel-wise manner and
    return the unique tuples of elements that do not match for the same spatial position.

    Args:
      l1 (np.ndarray): array 1
      l2 (np.ndarray): array 2

    Examples (supposing the following lists are np.ndarrays):
      - compare_pixelwise([1,2,3], [1,2,4]) → [[3], [4]]
      - compare_pixelwise([1,2,4,3], [1,2,3,5]) → [[3, 4], [5, 3]]

    Returns:
      np.ndarray: unique_diffs: 2D, with columns having the differences for the same position
      sorted in ascending order using the l1 elements
    """
    # assert all([isinstance(l1, np.ndarray), isinstance(l2, np.ndarray),
    #             l1.dtype == np.dtype(int), l2.dtype == np.dtype(int),
    #             l1.shape == l2.shape, l1.dtype == l2.dtype]), (
    #                 f'{type(l1)}, {type(l2)}, {l1.dtype}, {l2.dtype}, {l1.shape}, {l2.shape}, {l1}, {l2}')
    cond = l1 != l2
    uids_tuples = np.array([], dtype=l1.dtype)
    if np.any(cond):
        uids_tuples = np.unique(np.stack([l1[cond], l2[cond]]), axis=1)
    return uids_tuples


def parse__sid_pid2eid__v2(sid_pid2eid__template: Dict[Union[int, 'DEFAULT'], Union[int, 'IGNORED']]):
    """
    Parsing priority, sid_pid is mapped to:
      1. sid_pid2eid__template[sid_pid] if it exists, else
      2. sid_pid2eid__template[sid] if it exists, else
      3. sid_pid2eid__template['DEFAULT'] value

    Returns:
      sid_pid2eval_id: a dense mapping having keys for all possible sid_pid s (0 to 99_99)
        using the provided sparse sid_pid2eid__template and the reserved DEFAULT key and IGNORED value.
    """
    sp2e = sid_pid2eid__template
    sp2e_keys = sp2e.keys()
    sp2e_new = dict()
    for k in range(99_99):

        if k in sp2e_keys:
            sp2e_new[k] = sp2e[k]
            continue

        sid, pid = (k, None) if k < 100 else divmod(k, 100)
        if sid in sp2e_keys:
            sp2e_new[k] = sp2e[sid]
            continue

        if 'DEFAULT' in sp2e_keys:
            sp2e_new[k] = sp2e['DEFAULT']
            continue

        raise ValueError(f'sid_pid2eid__template does not follow the specification rules for key {k}.')

    # replace ignored sid_pid s with the correct ignored eval_id
    eval_id_max = max(filter(lambda v: isinstance(v, int), sp2e_new.values()))
    sp2e_new = {k: eval_id_max + 1 if v == 'IGNORED' else v for k, v in sp2e_new.items()}

    return sp2e_new


def UNUSED_parse_dataset_sid_pid2eval_sid_pid(dataset_sid_pid2eval_sid_pid, experimental_noinfo_id=0):
    """
    Parsing priority, sid_pid is mapped to:
      1. dataset_sid_pid2eval_sid_pid[sid_pid] if it exists, else
      2. dataset_sid_pid2eval_sid_pid[sid] if it exists, else
      3. dataset_sid_pid2eval_sid_pid['DEFAULT'] value

    Returns:
      sid_pid2eval_id: a dense mapping having keys for all possible sid_pid s (0 to 99_99)
        using the provided sparse dataset_sid_pid2eval_sid_pid
    """
    dsp2spe = copy.copy(dataset_sid_pid2eval_sid_pid)
    dsp2spe_keys = dsp2spe.keys()
    dsp2spe_new = dict()
    for k in range(10000):
        if k in dsp2spe_keys:
            dsp2spe_new[k] = dsp2spe[k]
            continue
        sid, pid = (k, None) if k < 100 else divmod(k, 100)
        if sid in dsp2spe_keys:
            dsp2spe_new[k] = dsp2spe[sid]
            continue
        if 'DEFAULT' in dsp2spe_keys:
            dsp2spe_new[k] = dsp2spe['DEFAULT']
            continue
        raise ValueError(f'dataset_sid_pid2eval_sid_pid does not follow the specification rules for key {k}.')
    assert all(v in list(range(10000)) + ['IGNORED'] for v in dsp2spe_new.values())
    # replace ignored sid_pid s with the experimental_noinfo_id
    dsp2spe_new = {k: experimental_noinfo_id if v == 'IGNORED' else v for k, v in dsp2spe_new.items()}
    return dsp2spe_new


def UNUSED_generate_ignore_info(panoptic_dict, panoptic_ann_img, image_id, void=0):
    # Create empty ignore_img and ignore_dict
    ignore_img = np.zeros_like(panoptic_ann_img).astype(np.uint8)
    ignore_dict = dict()

    # Get panoptic segmentation in the correct format
    pan_ann_format = panoptic_ann_img[..., 0] + panoptic_ann_img[..., 1] * 256 + panoptic_ann_img[..., 2] * 256 * 256

    # Store overall void info in ignore_img and ignore_dict
    overall_void = pan_ann_format == void
    ignore_img[overall_void] = 255
    ignore_dict['255'] = 255

    # Retrieve annotation corresponding to image_id
    annotation_dict = dict()
    for annotation in panoptic_dict['annotations']:
        if annotation['image_id'] == image_id:
            annotation_dict = annotation

    if len(annotation_dict) == 0:
        raise KeyError('ImageID is not present in the panoptic annotation dict.')

    # Find crowd annotations and add them to ignore_img and ignore_dict
    for inst_annotation in annotation_dict['segments_info']:
        if inst_annotation['iscrowd'] == 1:
            crowd_instance_id = inst_annotation['id']
            category_id = inst_annotation['category_id']
            crowd_mask = pan_ann_format == crowd_instance_id
            ignore_img[crowd_mask] = category_id
            ignore_dict[str(category_id)] = category_id

    return ignore_img[:, :, 0], ignore_dict


def get_filenames_in_dir(directory):
    filenames = [file for file in glob.glob(directory + "/*")]
    filenames.extend([file for file in glob.glob(directory + "/*/*")])
    return filenames


def find_filename_in_list(filename, filename_list, subject='', ext=None):
    f_found = None
    for fs in filename_list:
        if ext is not None:
            if filename in fs and fs.endswith(str(ext)):
                f_found = fs
        else:
            if filename in fs:
                f_found = fs

    if f_found is None:
        raise FileNotFoundError('There is no corresponding ' + str(subject) + ' prediction file for ' + filename)

    return f_found
