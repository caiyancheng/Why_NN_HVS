#!/bin/bash

# 激活conda环境（假设环境名为myenv）
# 注意：激活conda环境要用source并指定conda.sh路径
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hvs_2
cd /auto/homes/yc613/Py_codes/Why_NN_HVS/Train_on_ImageNet

python train_main.py \
    --train-dir /local/scratch-2/yc613/Datasets/ILSVRC2012_img_train \
    --val-dir /local/scratch-2/yc613/Datasets/ILSVRC2012_img_val \
    --model resnet18 \
    --epochs 60 \
    --batch-size 256 \
    --img-size 224 \
    --workers 12 \
    --opt adamw \
    --lr 1.6 \
    --weight-decay 0.05 \
    --label-smoothing 0.1 \
    --warmup-epochs 5 \
    --min-lr 1e-6 \
    --amp \
    --channels-last \
    --output ./runs/resnet18_run

