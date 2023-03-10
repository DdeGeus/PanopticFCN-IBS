# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PanopticFCN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import torch
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from data.cityscapes.cityscapes_panoptic_separated import register_all_cityscapes_panoptic
from data.cityscapes.dataset_mapper import CityscapesPanopticDatasetMapper

from data.cityscapes.cityscapes_panoptic_separated_sampling import register_all_cityscapes_panoptic_newsampling
from data.cityscapes.dataset_mapper_sampling import CityscapesPanopticDatasetMapperSampling

from data.mapillaryvistas.dataset_mapper import MapillaryVistasPanopticDatasetMapper
from data.mapillaryvistas.mapillary_vistas_panoptic_separated import register_all_mapillary_vistas_panoptic

from data.mapillaryvistas.dataset_mapper_sampling import MapillaryVistasPanopticDatasetMapperSampling
from data.mapillaryvistas.mapillary_vistas_panoptic_separated_sampling import register_all_mapillary_vistas_panoptic_sampling


from panopticfcn import add_panopticfcn_config, build_lr_scheduler
os.environ["NCCL_LL_THRESHOLD"] = "0"
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg", "mapillary_vistas_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.DATASETS.NAME == 'Cityscapes':
            if cfg.INPUT.NEW_SAMPLING:
                mapper = CityscapesPanopticDatasetMapperSampling(cfg)
            else:
                mapper = CityscapesPanopticDatasetMapper(cfg)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.DATASETS.NAME == 'Mapillary':
            if cfg.INPUT.NEW_SAMPLING:
                mapper = MapillaryVistasPanopticDatasetMapperSampling(cfg)
            else:
                mapper = MapillaryVistasPanopticDatasetMapper(cfg)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            return build_detection_train_loader(cfg)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panopticfcn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if comm.is_main_process():
        if cfg.SAVE_PREDICTIONS:
            save_dir = os.path.join(cfg.SAVE_DIR, cfg.SAVE_DIR_NAME)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

    # Convert batch size to internally used batch size (for crop sampling)
    if cfg.INPUT.NEW_SAMPLING:
        assert cfg.SOLVER.IMS_PER_BATCH % 2 == 0
        print("Initial ims per batch: {}".format(cfg.SOLVER.IMS_PER_BATCH))
        cfg.SOLVER.defrost()
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH // 2
        cfg.SOLVER.freeze()
        print("Used ims per batch: {}".format(cfg.SOLVER.IMS_PER_BATCH))

    # TODO: assert that batch size is compatible with IBS

    if cfg.DATASETS.NAME == 'Cityscapes':
        if cfg.INPUT.NEW_SAMPLING:
            register_all_cityscapes_panoptic_newsampling(cfg)
        else:
            register_all_cityscapes_panoptic(cfg)
    elif cfg.DATASETS.NAME == 'Mapillary':
        if cfg.INPUT.NEW_SAMPLING:
            register_all_mapillary_vistas_panoptic_sampling(cfg)
        else:
            register_all_mapillary_vistas_panoptic(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    if args.machine_rank == -1:
        machine_rank = int(os.environ['SLURM_PROCID'])
    else:
        machine_rank = args.machine_rank
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )