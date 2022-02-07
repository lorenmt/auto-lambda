"""
This module contains tools for handling dataset specifications.
"""
import copy
import platform
from typing import Union

version = platform.python_version()
if float(version[:3]) <= 3.6:
  raise EnvironmentError('At least Python 3.7 is needed for ordered dict functionality.')

from ruamel_yaml import YAML


class DatasetSpec(object):
  """
  This class creates a dataset specification from a YAML specification file, so properties
  in the specification are easily accessed. Moreover, it provides defaults and specification checking.

  Specification attribute fields:
    - l: list of str, the names of the scene-level semantic classes
    - l_things: list of str, the names of the scene-level things classes
    - l_stuff: list of str, the names of the scene-level stuff classes
    - l_parts: list of str, the names of the scene-level classes with parts
    - l_noparts: list of str, the names of the scene-level classes without parts
    - scene_class2part_classes: dict, mapping for scene-level class name to part-level class names,
        the ordering of elements in scene_class2part_classes.keys() and scene_class2part_classes.values()
        implicitly defines the sid and pid respectively, which can be retrieved with the functions below
    - sid2scene_class: dict, mapping from sid to scene-level semantic class name
    - sid2scene_color: dict, mapping from sid to scene-level semantic class color
    - sid_pid2scene_class_part_class: dict, mapping from sid_pid to a tuple of
        (scene-level class name, part-level class name)

  Specification attribute functions:
    - scene_class_from_sid(sid)
    - sid_from_scene_class(name)
    - part_classes_from_sid(sid)
    - part_classes_from_scene_class(name)
    - scene_color_from_scene_class(name)
    - scene_color_from_sid(sid)
    - scene_class_part_class_from_sid_pid(sid_pid)
    - sid_pid_from_scene_class_part_class(scene_name, part_name)

  Examples (from Cityscapes Panoptic Parts):
    - for the 'bus' scene-level class and the 'wheel' part-level class it holds:
      - 'bus' in l_things → True
      - 'bus' in l_parts → True
      - sid_from_scene_class('bus') → 28
      - scene_color_from_scene_class('bus') → [0, 60, 100]
      - part_classes_from_scene_class('bus') → ['UNLABELED', 'window', 'wheel', 'light', 'license plate', 'chassis']
      - sid_pid_from_scene_class_part_class('bus', 'wheel') → 2802

  Experimental (format/API may change):
    - l_allparts: list of str, a list of all parts in str with format f"{scene_class}-{part_class}",
      contains at position 0 the special 'UNLABELED' class

  Notes:
    - A special 'UNLABELED' semantic class is defined for the scene-level and part-level abstractions.
        This class must have sid/pid = 0 and is added by befault to the attributes of this class if
        it does not exist in yaml specification.
    - It holds that:
      - the special 'UNLABELED' class ∈ l, l_stuff, l_noparts
      - l = l_things ∪ l_stuff
      - l = l_parts ∪ l_noparts
    - sids are continuous and zero-based
    - iids do not need to be continuous
    - pids are continuous and zero-based per sid
  """
  def __init__(self, spec_path):
    """
    Args:
      spec_path: a YAML panoptic parts dataset specification
    """
    with open(spec_path) as fd:
      spec = YAML().load(fd)

    self._spec_version = spec['version']
    self._dataset_name = spec['name']
    # describes the semantic information layer
    self._scene_class2part_classes = spec['scene_class2part_classes']
    # describes the instance information layer
    self._scene_classes_with_instances = spec['scene_classes_with_instances']
    self._scene_class2color = spec.get('scene_class2color')
    if self._scene_class2color is None:
      raise ValueError(
          '"scene_class2color" in dataset_spec must be provided for now. '
          'In the future random color assignment will be implemented.')
    self._countable_pids_groupings = spec.get('countable_pids_groupings')

    self._extract_attributes()

  def _extract_attributes(self):
    self.dataset_name = self._dataset_name

    def _check_and_append_unlabeled(seq: Union[dict, list], unlabeled_dct=None):
      seq = copy.copy(seq)
      if 'UNLABELED' not in seq:
        if isinstance(seq, dict):
          seq_new = unlabeled_dct
          seq_new.update(seq)
        elif isinstance(seq, list):
          seq_new = ['UNLABELED'] + seq
      if list(seq_new)[0] != 'UNLABELED':
        raise ValueError(
            f'"UNLABELED" class exists in seq but not at position 0. seq: {seq}')
      return seq_new

    # check and append (if doesn't exist) the special UNLABELED key to
    # scene_class2part_classes and scene_class2color attributes
    self.scene_class2part_classes = _check_and_append_unlabeled(self._scene_class2part_classes,
                                                                {'UNLABELED': []})
    self.scene_class2part_classes = dict(
        zip(self.scene_class2part_classes.keys(),
            map(_check_and_append_unlabeled,
                self.scene_class2part_classes.values())))
    self.scene_class2color = _check_and_append_unlabeled(self._scene_class2color,
                                                         {'UNLABELED': [0, 0, 0]})

    # self.sid_pid2scene_class_part_class is a coarse mapping (not all 0-99_99 keys are present)
    # from sid_pid to Tuple(str, str), it contains sid_pid with format S, SS, S_PP, SS_PP
    # where S >= 0, SS >= 0, S_PP >= 1_01, SS_PP >= 10_01, and PP >= 1
    self.sid_pid2scene_class_part_class = dict()
    for sid, (scene_class, part_classes) in enumerate(self.scene_class2part_classes.items()):
      for pid, part_class in enumerate(part_classes):
        sid_pid = sid if pid == 0 else sid * 100 + pid
        self.sid_pid2scene_class_part_class[sid_pid] = (scene_class, part_class)
    self.scene_class_part_class2sid_pid = {
        v: k for k, v in self.sid_pid2scene_class_part_class.items()}

    self.l = list(self.scene_class2part_classes)
    self.l_things = self._scene_classes_with_instances
    self.l_stuff = list(set(self.l) - set(self.l_things))
    self.l_parts = list(filter(lambda k: len(self.scene_class2part_classes[k]) >= 2,
                               self.scene_class2part_classes))
    self.l_noparts = list(set(self.l) - set(self.l_parts))
    self.l_allparts = ['UNLABELED']
    for scene_class, part_classes in self.scene_class2part_classes.items():
      if scene_class == 'UNLABELED':
        continue
      for part_class in part_classes:
        if part_class == 'UNLABELED':
          continue
        self.l_allparts.append(f'{scene_class}-{part_class}')
    self.sid2scene_class = dict(enumerate(self.l))
    self.sid2scene_color = {sid: self.scene_class2color[name] for sid, name in self.sid2scene_class.items()}
    self.sid2part_classes = {sid: part_classes
                             for sid, part_classes in enumerate(self.scene_class2part_classes.values())}

    # self._sid_pid_file2sid_pid is a sparse mapping (not all 0-99_99 keys are present), with 
    # sid_pid s in the annotation files mapped to the official sid_pid s of the dataset.
    # This can be used to remove the part-level instance information layer
    # from the uids in the annotation files (this only applies to PASCAL Panoptic Parts for now).
    if self._countable_pids_groupings is not None:
      self._sid_pid_file2sid_pid = {k: k for k in self.sid_pid2scene_class_part_class}
      for scene_class, part_class2pids_grouping in self._countable_pids_groupings.items():
        sid = self.sid_from_scene_class(scene_class)
        for part_class, pids_file in part_class2pids_grouping.items():
          for pid_file in pids_file:
            assert pid_file != 0, 'Unhandled case (pid_file = 0), raise an issue to maintainers.'
            sid_pid_file = sid if pid_file == 0 else sid * 100 + pid_file
            self._sid_pid_file2sid_pid[sid_pid_file] = self.scene_class_part_class2sid_pid[(scene_class, part_class)]

  def sid_from_scene_class(self, name):
    return self.l.index(name)

  def scene_class_from_sid(self, sid):
    return self.l[sid]

  def scene_color_from_scene_class(self, name):
    return self._scene_class2color[name]

  def scene_color_from_sid(self, sid):
    return self.sid2scene_color[sid]
  
  def part_classes_from_sid(self, sid):
    return self.sid2part_classes[sid]

  def part_classes_from_scene_class(self, name):
    return self.scene_class2part_classes[name]

  def scene_class_part_class_from_sid_pid(self, sid_pid):
    return self.sid_pid2scene_class_part_class[sid_pid]

  def sid_pid_from_scene_class_part_class(self, scene_name, part_name):
    return self.scene_class_part_class2sid_pid[(scene_name, part_name)]


if __name__ == '__main__':
  spec = DatasetSpec('panoptic_parts/specs/dataset_specs/ppp_datasetspec.yaml')
  print(*sorted(filter(lambda t: t[0] != t[1],
                       spec._sid_pid_file2sid_pid.items())), sep='\n')
  # spec = DatasetSpec('panoptic_parts/specs/dataset_specs/cpp_datasetspec.yaml')
  breakpoint()
