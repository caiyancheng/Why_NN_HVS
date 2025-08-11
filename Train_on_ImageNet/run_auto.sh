#!/bin/bash

# 激活conda环境（假设环境名为myenv）
# 注意：激活conda环境要用source并指定conda.sh路径
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hvs_2
cd /auto/homes/yc613/Py_codes/Why_NN_HVS/Train_on_ImageNet

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=29501 \
  /auto/homes/yc613/Py_codes/Why_NN_HVS/Train_on_ImageNet/train_main.py \
  --train-dir /local/scratch-2/yc613/Datasets/ILSVRC2012_img_train \
  --val-dir   /local/scratch-2/yc613/Datasets/ILSVRC2012_img_val \
  --model mobilenet_v3_large --epochs 60 --batch-size 256 --amp --channels-last \
  --output ./runs_mobilenet_v3_large

