#!/usr/bin/env python3
"""
Single-GPU ImageNet trainer (PyTorch + AMP)
- 单卡训练；每个 epoch 保存权重 + last/best
- 自动把命令行参数保存到 output/config.json 便于复现实验
- 新增：日志保存到文件（同时打印到终端）、CSV 指标日志
- 修复：默认使用标准 ImageNet 数据增强（RandomResizedCrop + CenterCrop）
- 新增：可选 RandAugment 与 RandomErasing；可配置 LR 线性缩放基准批量
- 修复：AMP GradScaler 的未来弃用提示
"""
import argparse
import csv
import logging
import math
import os
# 建议在外部 shell 里设置 CUDA_VISIBLE_DEVICES，这里保留成单卡 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import random
import time
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ------------------------- Utilities -------------------------

def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: str | None):
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def is_main_process():
    # 单卡场景恒为 True
    return True


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def reduce_tensor(t: torch.Tensor):
    # 单卡无需规约
    return t


# ----------------------- Dataset helpers -----------------------

def build_transforms(img_size: int, use_randaugment: bool, reprob: float):
    # 标准 ImageNet 数据增强
    train_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        # num_ops=2, magnitude=9 是常见的一个设置
        train_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if reprob > 0:
        train_list.append(transforms.RandomErasing(p=reprob, scale=(0.02, 0.33), value='random'))

    train_tf = transforms.Compose(train_list)

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def maybe_limit_classes(imgfolder: datasets.ImageFolder, class_limit: int | None):
    if class_limit is None:
        return imgfolder
    kept_classes = imgfolder.classes[:class_limit]
    kept_idx = set(imgfolder.class_to_idx[c] for c in kept_classes)
    imgfolder.samples = [s for s in imgfolder.samples if s[1] in kept_idx]
    imgfolder.targets = [t for t in imgfolder.targets if t in kept_idx]
    old_to_new = {old: new for new, old in enumerate(sorted(list(kept_idx)))}
    imgfolder.samples = [(p, old_to_new[t]) for (p, t) in imgfolder.samples]
    imgfolder.targets = [old_to_new[t] for t in imgfolder.targets]
    imgfolder.classes = kept_classes
    imgfolder.class_to_idx = {c: i for i, c in enumerate(kept_classes)}
    return imgfolder


# ------------------------- Models -------------------------

def create_model(name: str, num_classes: int, compile_flag: bool):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=None, num_classes=num_classes)
    elif name == "resnet34":
        m = models.resnet34(weights=None, num_classes=num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if compile_flag and hasattr(torch, "compile"):
        try:
            m = torch.compile(m)
        except Exception:
            pass
    return m


# ------------------------- Train / Eval -------------------------

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, loss_fn, scheduler=None, channels_last=False, logger=None):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    start = time.time()

    for i, (images, targets) in enumerate(tqdm(loader, total=len(loader), desc=f"Train {epoch+1}")):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            output = model(images)
            loss = loss_fn(output, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        acc1 = accuracy(output, targets, topk=(1,))[0]
        loss_red = reduce_tensor(loss.detach())
        acc1_red = reduce_tensor(acc1.detach())

        loss_meter.update(loss_red.item(), images.size(0))
        top1_meter.update(acc1_red.item(), images.size(0))

    elapsed = time.time() - start
    return loss_meter.avg, top1_meter.avg, elapsed


def validate(model, loader, device, loss_fn):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(loader, total=len(loader), desc="Validate"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                output = model(images)
                loss = loss_fn(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

            loss_red = reduce_tensor(loss.detach())
            acc1_red = reduce_tensor(acc1.detach())
            acc5_red = reduce_tensor(acc5.detach())

            loss_meter.update(loss_red.item(), images.size(0))
            top1_meter.update(acc1_red.item(), images.size(0))
            top5_meter.update(acc5_red.item(), images.size(0))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


# ------------------------- LR Schedules -------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_step = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.last_step += 1
        for i, g in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.last_step <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * self.last_step / self.warmup_steps
            else:
                t = (self.last_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * t))
            g["lr"] = lr


# ------------------------- Main -------------------------

def main():
    p = argparse.ArgumentParser(description="Single-GPU ImageNet Trainer")
    p.add_argument('--train-dir', type=str, default='/local/scratch-2/yc613/Datasets/ILSVRC2012_img_train')
    p.add_argument('--val-dir', type=str, default='/local/scratch-2/yc613/Datasets/ILSVRC2012_img_val')
    p.add_argument('--model', type=str, default='resnet18', choices=['mobilenet_v3_large','resnet18','resnet34','efficientnet_v2_s'])
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--opt', type=str, default='sgd', choices=['sgd','adamw'])
    p.add_argument('--lr', type=float, default=0.1, help='LR at lr-base-batch size')
    p.add_argument('--lr-base-batch', type=int, default=256, help='Batch size that lr refers to for linear scaling')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--class-limit', type=int, default=None)
    p.add_argument('--output', type=str, default='./runs')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log-file', type=str, default=None, help='log file path. Default: output/train.log')
    p.add_argument('--csv-log', type=str, default=None, help='CSV metrics log path. Default: output/metrics.csv')
    p.add_argument('--aa', action='store_true', help='Use RandAugment(num_ops=2, magnitude=9) during training')
    p.add_argument('--reprob', type=float, default=0.0, help='RandomErasing probability (0 to disable)')
    args = p.parse_args()

    # 输出目录 & 配置
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.log_file is None:
        args.log_file = str(Path(args.output) / 'train.log')
    if args.csv_log is None:
        args.csv_log = str(Path(args.output) / 'metrics.csv')

    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 日志
    logger = setup_logger(args.log_file)
    logger.info(f"Args: {args}")

    # 设备与性能开关
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    setup_seed(42)

    # Data
    train_tf, val_tf = build_transforms(args.img_size, use_randaugment=args.aa, reprob=args.reprob)
    train_set = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_set   = datasets.ImageFolder(args.val_dir, transform=val_tf)

    if args.class_limit is not None:
        train_set = maybe_limit_classes(train_set, args.class_limit)
        val_set   = maybe_limit_classes(val_set, args.class_limit)

    num_classes = len(train_set.classes)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Model
    model = create_model(args.model, num_classes=num_classes, compile_flag=args.compile).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Loss & Optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率线性缩放（单卡：global_batch = batch_size）
    global_batch = args.batch_size
    for g in optimizer.param_groups:
        g['lr'] = args.lr * (global_batch / max(1, args.lr_base_batch))
    eff_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Effective LR after scaling: {eff_lr:.6f} (base lr={args.lr}, base batch={args.lr_base_batch}, actual batch={global_batch})")

    total_steps = math.ceil(len(train_set) / global_batch) * args.epochs
    warmup_steps = math.ceil(len(train_set) / global_batch) * args.warmup_epochs
    scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=args.min_lr)

    # 使用新的 AMP API，消除弃用提示
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # 可选恢复
    start_epoch = 0
    best_top1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0)
        best_top1 = checkpoint.get('best_top1', 0.0)
        logger.info(f"Resumed from {args.resume}: epoch {start_epoch}, best@1 {best_top1:.2f}")

    # CSV 日志
    with open(args.csv_log, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["epoch","train_loss","train_top1","val_loss","val_top1","val_top5","epoch_minutes","lr"])

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_top1, tr_time = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, loss_fn, scheduler, channels_last=args.channels_last, logger=logger
        )
        val_loss, val_top1, val_top5 = validate(model, val_loader, device, loss_fn)

        msg = (
            f"Epoch {epoch+1}/{args.epochs} | train loss {tr_loss:.3f} top1 {tr_top1:.2f} | "
            f"val loss {val_loss:.3f} top1 {val_top1:.2f} top5 {val_top5:.2f} | "
            f"epoch time {tr_time/60:.1f} min | lr {optimizer.param_groups[0]['lr']:.6f}"
        )
        logger.info(msg)

        # 追加到 CSV
        with open(args.csv_log, 'a', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow([epoch+1, f"{tr_loss:.6f}", f"{tr_top1:.4f}", f"{val_loss:.6f}", f"{val_top1:.4f}", f"{val_top5:.4f}", f"{tr_time/60:.3f}", f"{optimizer.param_groups[0]['lr']:.6f}"])

        # 保存 checkpoint（每个 epoch）
        payload = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_top1': max(best_top1, val_top1),
            'args': vars(args),
        }
        ckpt_dir = Path(args.output)
        torch.save(payload, ckpt_dir / f'epoch_{epoch+1}.pth')
        torch.save(payload, ckpt_dir / 'last.pth')
        if val_top1 > best_top1:
            best_top1 = val_top1
            torch.save(payload, ckpt_dir / 'best.pth')

    logger.info(f"Training done. Best top-1: {best_top1:.2f}")


if __name__ == "__main__":
    main()

