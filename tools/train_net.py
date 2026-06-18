#!/usr/bin/env python
# encoding: utf-8

import sys

sys.path.append(".")

from fastreid.config import get_cfg
from fastreid.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from fastreid.utils.checkpoint import Checkpointer


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        return DefaultTrainer.test(cfg, model)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
