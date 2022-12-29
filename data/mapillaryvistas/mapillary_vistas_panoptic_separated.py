# Copyright (c) Facebook, Inc. and its affiliates.
## Adapted from detectron2/data/datasets/cityscapes_panoptic.py

import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Mapillary Vistas panoptic dataset to the DatasetCatalog.
"""

logger = logging.getLogger(__name__)

def get_mapillary_panoptic_files(image_dir, gt_dir, json_info):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    for basename in PathManager.ls(image_dir):
        image_file = os.path.join(image_dir, basename)

        suffix = ".jpg"
        assert basename.endswith(suffix), basename
        basename = os.path.basename(basename)[: -len(suffix)]

        image_dict[basename] = image_file

    for ann in json_info["annotations"]:
        image_file = image_dict.get(ann["image_id"], None)
        assert image_file is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        files.append((image_file, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files


def load_mapillary_vistas_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_mapillary_panoptic_files(image_dir, gt_dir, json_info)
    ret = []

    for image_file, label_file, segments_info in files:
        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append(
            {
                "file_name": image_file,
                "image_id": os.path.basename(image_file).replace(".jpg",""),
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa

    return ret


_RAW_MAPILLARY_VISTAS_PANOPTIC_SPLITS = {
    "mapillary_vistas_panoptic_train_separated": (
        "mapillary-vistas/training/images",
        "mapillary-vistas/training/panoptic",
        "mapillary-vistas/training/panoptic/panoptic_2018.json",
    ),
    "mapillary_vistas_panoptic_val_separated": (
        "mapillary-vistas/validation/images",
        "mapillary-vistas/validation/panoptic",
        "mapillary-vistas/validation/panoptic/panoptic_2018.json",
    ),
}


def register_all_mapillary_vistas_panoptic(cfg):
    root = os.getenv("DETECTRON2_DATASETS", "datasets")

    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    images_dir = _RAW_MAPILLARY_VISTAS_PANOPTIC_SPLITS["mapillary_vistas_panoptic_train_separated"][0]
    mapillary_categories_json = images_dir.replace("training/images", "categories_mapillary_2018.json")
    mapillary_categories_json = os.path.join(root, mapillary_categories_json)
    with open(mapillary_categories_json) as fp:
        MAPILLARY_CATEGORIES = json.load(fp)

    thing_classes = [k["name"] for k in MAPILLARY_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in MAPILLARY_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in MAPILLARY_CATEGORIES if k["isthing"] == 0]
    stuff_colors = [k["color"] for k in MAPILLARY_CATEGORIES if k["isthing"] == 0]

    for k in MAPILLARY_CATEGORIES:
        k["trainId"] = k["id"] - 1

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    contiguous_id_to_thing_train_id = dict()
    contiguous_id_to_stuff_train_id = dict()
    thing_id = 0
    stuff_id = 0
    contiguous_id_to_dataset_id = {}

    for k in MAPILLARY_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
            contiguous_id_to_thing_train_id[k["trainId"]] = thing_id
            thing_id += 1
            if cfg.MODEL.POSITION_HEAD.STUFF.ALL_CLASSES:
                stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
                contiguous_id_to_stuff_train_id[k["trainId"]] = stuff_id
                stuff_id += 1
        else:
            if cfg.MODEL.POSITION_HEAD.STUFF.WITH_THING and not cfg.MODEL.POSITION_HEAD.STUFF.ALL_CLASSES:
                stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
                contiguous_id_to_stuff_train_id[k["trainId"]] = stuff_id + 1
                stuff_id += 1
            else:
                stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
                contiguous_id_to_stuff_train_id[k["trainId"]] = stuff_id
                stuff_id += 1
        contiguous_id_to_dataset_id[k["trainId"]] = k["id"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_thing_train_id"] = contiguous_id_to_thing_train_id
    meta["contiguous_id_to_stuff_train_id"] = contiguous_id_to_stuff_train_id
    meta["contiguous_id_to_dataset_id"] = contiguous_id_to_dataset_id

    thing_train_id2contiguous_id = dict(
        zip(contiguous_id_to_thing_train_id.values(), contiguous_id_to_thing_train_id.keys()))

    stuff_train_id2contiguous_id = dict(
        zip(contiguous_id_to_stuff_train_id.values(), contiguous_id_to_stuff_train_id.keys()))

    meta["thing_train_id2contiguous_id"] = thing_train_id2contiguous_id
    meta["stuff_train_id2contiguous_id"] = stuff_train_id2contiguous_id

    print(meta)

    for key, (image_dir, gt_dir, gt_json) in _RAW_MAPILLARY_VISTAS_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_mapillary_vistas_panoptic(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir.replace("mapillary_vistas_panoptic_", ""),
            evaluator_type="mapillary_vistas_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )
