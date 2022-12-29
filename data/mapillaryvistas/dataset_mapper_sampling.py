# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted from detectron2/data/dataset_mapper.py

import copy
import os
import json
import sys
import logging
import numpy as np
from typing import Callable, List, Union
import torch
import pycocotools

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

from data.mapillaryvistas.augmentations import RandomCropWithInstance, AugInput, ResizeShortestEdgeAdaptive

__all__ = ["MapillaryVistasPanopticDatasetMapperSampling"]


class MapillaryVistasPanopticDatasetMapperSampling:
  """
  A callable which takes a dataset dict in Detectron2 Dataset format,
  and map it into a format used by the model.

  This is the default callable to be used to map your dataset dict into training data.
  You may need to follow it to implement your own one for customized logic,
  such as a different way to read or transform images.
  See :doc:`/tutorials/data_loading` for details.

  The callable currently does the following:

  1. Read the image from "file_name"
  2. Applies cropping/geometric transforms to the image and annotations
  3. Prepare data and annotations to Tensor and :class:`Instances`
  """

  @configurable
  def __init__(
      self,
      cfg,
      is_train: bool,
      *,
      augmentations: List[Union[T.Augmentation, T.Transform]],
      image_format: str,
      images_per_class_json: str,
      use_instance_mask: bool = False,
      instance_mask_format: str = "polygon",
      recompute_boxes: bool = False
  ):
    """
    NOTE: this interface is experimental.

    Args:
        cfg: config dict
        is_train: whether it's used in training or inference
        augmentations: a list of augmentations or deterministic transforms to apply
        image_format: an image format supported by :func:`detection_utils.read_image`.
        use_instance_mask: whether to process instance segmentation annotations, if available
        instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
            masks into this format.
        recompute_boxes: whether to overwrite bounding box annotations
            by computing tight bounding boxes from instance mask annotations.
    """
    if recompute_boxes:
      assert use_instance_mask, "recompute_boxes requires instance masks"
    # fmt: off
    self.cfg = cfg
    dataset_names = self.cfg.DATASETS.TRAIN
    self.meta = MetadataCatalog.get(dataset_names[0])
    self.is_train = is_train
    self.augmentations = T.AugmentationList(augmentations)
    self.image_format = image_format
    self.use_instance_mask = use_instance_mask
    self.instance_mask_format = instance_mask_format
    self.recompute_boxes = recompute_boxes
    self.images_per_class_json = images_per_class_json
    # fmt: on
    logger = logging.getLogger(__name__)
    mode = "training" if is_train else "inference"
    logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
    with open(self.images_per_class_json, 'r') as fp:
      self.images_per_class_dict = json.load(fp)
    self.num_images_per_class = dict()
    for cat_id in self.images_per_class_dict['cat_ids']:
      self.num_images_per_class[cat_id] = len(self.images_per_class_dict['imgs_per_cat'][str(cat_id)])
    print("self.num_images_per_class", self.num_images_per_class)
    self.scale_options = self.cfg.INPUT.NEW_SAMPLING_SCALES

  @classmethod
  def from_config(cls, cfg, is_train: bool = True):
    augs = utils.build_augmentation(cfg, is_train)
    if cfg.INPUT.NEW_SAMPLING and is_train:
      augs[0] = ResizeShortestEdgeAdaptive(cfg.INPUT.MIN_SIZE_TRAIN,
                                           cfg.INPUT.MAX_SIZE_TRAIN,
                                           cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                                           scale_min=cfg.INPUT.NEW_SAMPLING_SCALE_MIN,
                                           scale_max=cfg.INPUT.NEW_SAMPLING_SCALE_MAX,
                                           overall_max=cfg.INPUT.NEW_SAMPLING_OVERALL_MAX)
    if cfg.INPUT.CROP.ENABLED and is_train:
      if cfg.INPUT.CROP.MINIMUM_INST_AREA == 0:
        augs.insert(1, T.RandomCrop(cfg.INPUT.CROP.TYPE,
                                    cfg.INPUT.CROP.SIZE))
      else:
        assert cfg.INPUT.CROP.MINIMUM_INST_AREA > 0
        augs.insert(1, RandomCropWithInstance(cfg.INPUT.CROP.TYPE,
                                              cfg.INPUT.CROP.SIZE))
      recompute_boxes = cfg.MODEL.MASK_ON
    else:
      recompute_boxes = False

    dataset_names = cfg.DATASETS.TRAIN
    meta = MetadataCatalog.get(dataset_names[0])

    ret = {
      "cfg": cfg,
      "is_train": is_train,
      "augmentations": augs,
      "image_format": cfg.INPUT.FORMAT,
      "images_per_class_json": meta.images_per_class_json,
      "use_instance_mask": cfg.MODEL.MASK_ON,
      "instance_mask_format": cfg.INPUT.MASK_FORMAT,
      "recompute_boxes": recompute_boxes,
    }

    return ret

  @staticmethod
  def _convert_category_id(cat_id, meta):
    if cat_id in meta.thing_dataset_id_to_contiguous_id:
      cat_id_converted = meta.thing_dataset_id_to_contiguous_id[cat_id]
    else:
      cat_id_converted = meta.stuff_dataset_id_to_contiguous_id[cat_id]
    return cat_id_converted

  @staticmethod
  def is_thing(cat_id, meta):
    return cat_id in meta.thing_dataset_id_to_contiguous_id.values()

  def __call__(self, input_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """

    output_dict = dict()
    output_dict['height'] = self.cfg.INPUT.CROP.SIZE[0]
    output_dict['width'] = self.cfg.INPUT.CROP.SIZE[1]

    input_dict = copy.deepcopy(input_dict)  # it will be modified by code below
    if self.is_train:
      num_ims = 2

      # Select a scale from the options
      scale_rand_id = np.random.randint(len(self.scale_options))
      scale_selected = self.scale_options[scale_rand_id]

    else:
      num_ims = 1

    for im_i in range(num_ims):
      if self.is_train:
        # Retrieve the randomly sampled category id
        cat_id = copy.deepcopy(input_dict['cat_id'])

        # Get image_id for this category_id
        num_images_per_cat = copy.deepcopy(self.num_images_per_class[cat_id])
        image_rand_id = np.random.randint(num_images_per_cat)

        # Retrieve dataset_dict for this image_id
        image_id = self.images_per_class_dict['imgs_per_cat'][str(cat_id)][image_rand_id]
        sample_json = os.path.join(self.meta.json_per_img_dir, image_id + '.json')
        with open(sample_json, 'r') as fp:
          dataset_dict = json.load(fp)

        # convert category_ids to contiguous ids
        for segm_info in dataset_dict['segments_info']:
          segm_info['category_id'] = self._convert_category_id(segm_info['category_id'], self.meta)
        cat_id = self._convert_category_id(copy.deepcopy(cat_id), self.meta)

        # Sample a random instance or segment from the cat_id
        segments_per_cat = list()
        for segm_info in dataset_dict['segments_info']:
          if segm_info['category_id'] == cat_id:
            segments_per_cat.append(segm_info)
        num_segments_per_cat = len(segments_per_cat)
        segm_rand_id = np.random.randint(num_segments_per_cat)
        segm_selected = segments_per_cat[segm_rand_id]

        # Get the bounding box from that particular instance
        # bbox is in format XYWH_ABS
        segm_bbox = segm_selected['bbox']
        segm_bbox_area = np.sqrt(segm_bbox[2] * segm_bbox[3])

        # Calculate rescale ratio
        if self.is_thing(cat_id, self.meta):
          if self.cfg.INPUT.NEW_SAMPLING_APPLY_SCALING:
            im_rescale_ratio = scale_selected/segm_bbox_area
            im_rescale_ratio = np.array(im_rescale_ratio).astype(np.float32)
          else:
            im_rescale_ratio = None
        else:
          im_rescale_ratio = None
      else:
        im_rescale_ratio = None
        dataset_dict = input_dict

      # USER: Write your own image loading if it's not from a file
      image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
      utils.check_image_size(dataset_dict, image)

      things_classes = list(self.meta.thing_dataset_id_to_contiguous_id.values())
      stuff_classes = list(self.meta.stuff_dataset_id_to_contiguous_id.values())

      # USER: Remove if you don't do semantic/panoptic segmentation.
      if "pan_seg_file_name" in dataset_dict:
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"))
        pan_seg_gt = pan_seg_gt[:, :, 0] + 256 * pan_seg_gt[:, :, 1] + 256 * 256 * pan_seg_gt[:, :, 2]
      else:
        raise NotImplementedError("Currently only possible if pan seg GT image file name is given")
        # pan_seg_gt = None

      # inst_map = np.zeros_like(pan_seg_gt, dtype=np.uint8)
      # Create annotations in desired instance segmentation format
      if self.cfg.INPUT.NEW_SAMPLING and self.is_train:
        boxes = [segm_bbox]
        print("boxes", boxes)
      else:
        boxes = list()
      annotations = list()

      for segment in dataset_dict['segments_info']:
        if segment['category_id'] in things_classes:
          annotation = dict()
          annotation['id'] = segment['id']
          annotation['bbox'] = segment['bbox']
          if self.cfg.INPUT.NEW_SAMPLING and self.is_train:
            pass
          else:
            boxes.append(segment['bbox'])
          annotation['bbox_mode'] = BoxMode.XYWH_ABS
          annotation['category_id'] = self.meta.contiguous_id_to_thing_train_id[segment['category_id']]
          annotation['iscrowd'] = segment['iscrowd']
          annotations.append(annotation)

      mapper = np.ones([100]).astype(np.uint8) * self.meta.ignore_label
      for i in range(len(mapper)):
        if i in self.meta.stuff_dataset_id_to_contiguous_id.keys():
          cont_id = self.meta.stuff_dataset_id_to_contiguous_id[i]
          stuff_id = self.meta.contiguous_id_to_stuff_train_id[cont_id]

          mapper[i] = stuff_id
        elif i in self.meta.thing_dataset_id_to_contiguous_id.keys():
          mapper[i] = 0

      boxes = np.array(boxes)
      if not len(boxes) == 0:
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

      aug_input = AugInput(image,
                           pan_seg=pan_seg_gt.astype(np.float32),
                           boxes=boxes,
                           ratio=im_rescale_ratio)
      transforms = self.augmentations(aug_input)
      image, pan_seg_gt = aug_input.image, aug_input.pan_seg.astype(np.int32)

      pan_seg_unique = np.unique(pan_seg_gt)

      sem_seg_gt = np.ones_like(pan_seg_gt).astype(np.uint8) * self.meta.ignore_label
      for segment in dataset_dict['segments_info']:
        if segment['id'] in pan_seg_unique:
          if segment['category_id'] in stuff_classes:
            mask = (pan_seg_gt == segment['id'])
            sem_seg_gt[mask] = self.meta.contiguous_id_to_stuff_train_id[segment['category_id']]

      annotations_filtered = list()
      for anno in annotations:
        if anno['id'] in pan_seg_unique:
          mask = (pan_seg_gt == anno['id'])
          anno['segmentation'] = pycocotools.mask.encode(np.asarray(mask, order="F"))
          annotations_filtered.append(anno)
          sem_seg_gt[mask] = 0
      if len(annotations) > 0:
        dataset_dict['annotations'] = annotations_filtered

      image_shape = image.shape[:2]  # h, w
      # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
      # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
      # Therefore it's important to use torch.Tensor.
      dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
      if sem_seg_gt is not None:
        dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

      if not self.is_train:
        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        return dataset_dict

      if "annotations" in dataset_dict:
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
          if not self.use_instance_mask:
            anno.pop("segmentation", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
          obj
          for obj in dataset_dict.pop("annotations")
          if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
          annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.

        if self.recompute_boxes and len(instances) > 0:
          instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if len(dataset_dict['instances']) == 0:
          del dataset_dict["instances"]

      output_dict[im_i] = dataset_dict

    return output_dict