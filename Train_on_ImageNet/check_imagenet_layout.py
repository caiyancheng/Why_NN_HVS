#!/usr/bin/env python3
import os
from torchvision import datasets, transforms

def summarize(name, ds):
    print(f"[{name}] classes: {len(ds.classes)}  images: {len(ds.samples)}")
    print(f"[{name}] first 5 classes: {ds.classes[:5]}")
    # 统计每类样本是否>=1（val 常见问题：某些类空了）
    per_class = {i:0 for i in range(len(ds.classes))}
    for _, y in ds.samples[:min(10000, len(ds.samples))]:  # 抽样检查前1万张
        per_class[y]+=1
    missing = [i for i,cnt in per_class.items() if cnt==0]
    if missing:
        print(f"[WARN] {name} has classes with 0 images (showing up to 10):",
              [ds.classes[i] for i in missing[:10]])
    else:
        print(f"[{name}] sampled check: every class has images (good).")

def main(
    train_dir="/local/scratch-2/yc613/Datasets/ILSVRC2012_img_train",
    val_dir="/local/scratch-2/yc613/Datasets/ILSVRC2012_img_val_sorted",
    img_size=224,
):
    tf = transforms.Compose([transforms.Resize((img_size, img_size))])

    train_ds = datasets.ImageFolder(train_dir, transform=tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=tf)

    summarize("TRAIN", train_ds)
    summarize("VAL  ", val_ds)

    # 1) 类别数是否相同
    if len(train_ds.classes) != len(val_ds.classes):
        print(f"[ERROR] class count mismatch: train={len(train_ds.classes)} val={len(val_ds.classes)}")
        return

    # 2) 类别名（文件夹名）顺序是否一致（必须一致！）
    if train_ds.classes != val_ds.classes:
        print("[ERROR] class name order mismatch between train and val.")
        # 找出不一样的地方（列几个例子）
        for i, (ct, cv) in enumerate(zip(train_ds.classes, val_ds.classes)):
            if ct != cv:
                print(f"  at index {i}: train='{ct}' vs val='{cv}'")
                break
        # 给出修复建议
        print("建议：val_sorted 的文件夹名（synset）要和 train 完全一致。检查你整理 val 的脚本。")
        return
    else:
        print("[OK] class names/order match between train and val.")

    # 3) class_to_idx 映射是否一致
    if train_ds.class_to_idx != val_ds.class_to_idx:
        print("[ERROR] class_to_idx mapping differs between train and val.")
        return
    else:
        print("[OK] class_to_idx mapping is identical.")

    # 4) 样本数量粗检（ImageNet-1K 通常约 1,281,167 train / 50,000 val）
    if len(val_ds.samples) != 50000:
        print(f"[WARN] val images not 50k (got {len(val_ds.samples)}). 确认是否少拷贝/移动。")
    if len(train_ds.samples) < 1200000:
        print(f"[WARN] train images unusually low: {len(train_ds.samples)}")

    print("\nAll checks done.")

if __name__ == "__main__":
    # 如需自定义目录，改这里或用 argparse
    main()
