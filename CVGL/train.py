#!/usr/bin/env python3
"""Train one branch of the PhotoMappers cross-view geo-localization model
(Sample4Geo contrastive training with a DINOv3-L backbone).

Two branches are trained separately; their similarity matrices are fused afterwards in predict.py:

  --task vgi_rsi   query = VGI photo  (384x384)  <->  reference = RSI satellite tile (384x384),  lr 1e-4
  --task svi_vgi   query = SVI panorama (336x672) <->  reference = VGI photo (336x336),          lr 1e-5,
                   dynamic position embedding (the shared ViT sees two different input shapes)

Every preset value can be overridden on the command line.  Both branches start from the DINOv3
self-supervised backbone (timm 'vit_large_patch16_dinov3', weights downloaded from the HF hub).
Recipe used in the paper: batch 32, 40 epochs, cosine schedule with 1 warm-up epoch, AdamW,
InfoNCE with label smoothing 0.1, AMP, gradient checkpointing, random (not GPS/similarity) sampling.

The test split (splits/val-19zl.csv) is evaluated every 4 epochs and a checkpoint
weights_e<epoch>_<R@1>.pth is written whenever R@1 improves (Sample4Geo default); predict.py picks
the file with the highest R@1.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --task vgi_rsi --data data/cvformat/disaster_vgi
  CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task svi_vgi --data data/cvformat/disaster_svi2vgi
"""
import os
import sys
import json
import time
import shutil
import argparse
import torch
from dataclasses import dataclass, asdict
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrain
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.cvusa_and_cvact import evaluate
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


PRESETS = {
    "vgi_rsi": dict(img_size=384, ground_hw=(384, 384), dynamic_img_size=False, lr=1e-4),
    "svi_vgi": dict(img_size=336, ground_hw=(336, 672), dynamic_img_size=True, lr=1e-5),
}


@dataclass
class Configuration:

    # Model
    model: str = 'vit_large_patch16_dinov3'

    # Image sizes: reference is img_size x img_size, query (ground) is ground_hw = (H, W)
    img_size: int = 384
    ground_hw: tuple = (384, 384)
    dynamic_img_size: bool = False

    # Training
    mixed_precision: bool = True
    seed: int = 42
    epochs: int = 40
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0, 1, 2, 3)   # GPU ids for training

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad: float = 100.            # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = True    # Gradient Checkpointing

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 1e-4
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"

    # Dataset (CVUSA-format tree written by build_dataset.py)
    data_folder: str = "data/cvformat/disaster_vgi"

    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously

    # Output directory for checkpoints and log
    model_path: str = "checkpoints/vgi_rsi"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from (weights only, strict=False); None = DINOv3 SSL backbone
    checkpoint_start: str = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(PRESETS), required=True)
    ap.add_argument("--data", required=True, help="CVUSA-format tree, e.g. data/cvformat/disaster_vgi")
    ap.add_argument("--out", default=None, help="checkpoint dir (default checkpoints/<task>)")
    ap.add_argument("--gpus", default="0,1,2,3",
                    help="DataParallel device ids (pick physical GPUs with CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--model", default=Configuration.model)
    ap.add_argument("--img-size", type=int, help="reference size (preset: 384 / 336)")
    ap.add_argument("--ground-hw", help="query size H,W (preset: 384,384 / 336,672)")
    ap.add_argument("--lr", type=float, help="peak learning rate (preset: 1e-4 / 1e-5)")
    ap.add_argument("--epochs", type=int, default=Configuration.epochs)
    ap.add_argument("--batch-size", type=int, default=Configuration.batch_size)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--checkpoint-start", default=None, help="warm-start from these weights")
    a = ap.parse_args()

    p = PRESETS[a.task]
    return Configuration(
        model=a.model,
        img_size=a.img_size or p["img_size"],
        ground_hw=tuple(int(x) for x in a.ground_hw.split(",")) if a.ground_hw else p["ground_hw"],
        dynamic_img_size=p["dynamic_img_size"],
        epochs=a.epochs,
        batch_size=a.batch_size,
        gpu_ids=tuple(int(x) for x in a.gpus.split(",")),
        grad_checkpointing=not a.no_grad_ckpt,
        lr=a.lr or p["lr"],
        data_folder=a.data,
        model_path=a.out or f"checkpoints/{a.task}",
        checkpoint_start=a.checkpoint_start,
    )


if __name__ == '__main__':

    config = parse_args()

    model_path = config.model_path

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))
    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size,
                      dynamic_img_size=config.dynamic_img_size)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)
    img_size_ground = tuple(config.ground_hw)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start, weights_only=False)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )


    # Train
    train_dataset = CVUSADatasetTrain(data_folder=config.data_folder ,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )


    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=True,
                                  pin_memory=True)


    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )


    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)



    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",
                                          transforms=ground_transforms_val,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)


    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))


    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))


    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))


        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0


    for epoch in range(1, config.epochs+1):

        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))


        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
