#!/usr/bin/env python3
import argparse, os, shutil, sys
from pathlib import Path
from scipy.io import loadmat

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x  # fallback

def build_id_to_wnid(meta_mat_path: str) -> dict[int, str]:
    """Return mapping: ILSVRC2012_ID (1..1000) -> WNID (e.g., 'n01440764')."""
    meta = loadmat(meta_mat_path, squeeze_me=True)['synsets']
    # 'synsets' is an array of structs; each has fields like 'ILSVRC2012_ID' and 'WNID'
    id2wnid = {}
    for s in meta:
        ilsvrc_id = int(s['ILSVRC2012_ID'])
        wnid = str(s['WNID'])
        if 1 <= ilsvrc_id <= 1000:
            id2wnid[ilsvrc_id] = wnid
    if len(id2wnid) != 1000:
        raise RuntimeError(f"Expected 1000 classes, got {len(id2wnid)} from meta.mat")
    return id2wnid

def read_val_gt(gt_path: str) -> list[int]:
    with open(gt_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if len(labels) != 50000:
        print(f"[Warn] Expected 50000 gt labels, got {len(labels)}", file=sys.stderr)
    return labels

def main():
    ap = argparse.ArgumentParser(description="Sort ImageNet val/ into ImageFolder layout using devkit.")
    ap.add_argument('--val-dir', default='/local/scratch-2/yc613/Datasets/ILSVRC2012_img_val', help="Directory with flat val images (e.g., ILSVRC2012_val_*.JPEG)")
    ap.add_argument('--devkit-dir', default='/local/scratch-2/yc613/Datasets/ILSVRC2012_devkit_t12', help="Path to ILSVRC2012_devkit_t12 directory")
    ap.add_argument('--out-dir', default='/local/scratch-2/yc613/Datasets/ILSVRC2012_img_val_sorted', help="Output directory to create ImageFolder-style val")
    ap.add_argument('--mode', choices=['copy','move','symlink'], default='copy', help="How to place files into out-dir")
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    devkit_dir = Path(args.devkit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    meta_path = devkit_dir / "data" / "meta.mat"
    if not gt_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing devkit files: {gt_path} or {meta_path}")

    id2wnid = build_id_to_wnid(str(meta_path))
    labels = read_val_gt(str(gt_path))

    # Collect val images in official order: ILSVRC2012_val_00000001.JPEG ... 00050000.JPEG
    imgs = sorted(val_dir.glob("ILSVRC2012_val_*.JPEG"))
    if len(imgs) != len(labels):
        print(f"[Warn] #images ({len(imgs)}) != #labels ({len(labels)}). Proceeding with min length.", file=sys.stderr)
    N = min(len(imgs), len(labels))

    def place(src: Path, dst: Path, mode: str):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == 'copy':
            if not dst.exists(): shutil.copy2(src, dst)
        elif mode == 'move':
            if dst.exists(): dst.unlink()
            shutil.move(str(src), str(dst))
        else:  # symlink
            if dst.exists(): return
            os.symlink(src.resolve(), dst)

    for i in tqdm(range(N), desc=f"Sorting ({args.mode})"):
        img = imgs[i]
        cls_id = labels[i]                 # 1..1000
        wnid = id2wnid[cls_id]             # e.g., 'n01440764'
        dst = out_dir / wnid / img.name
        place(img, dst, args.mode)

    print(f"Done. ImageFolder val is at: {out_dir}")

if __name__ == "__main__":
    main()
