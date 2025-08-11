#!/bin/bash

# 激活conda环境（假设环境名为myenv）
# 注意：激活conda环境要用source并指定conda.sh路径
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hvs_2
cd /auto/homes/yc613/Py_codes/Why_NN_HVS/Train_on_ImageNet

python train_main.py --model mobilenet_v3_large --output ./runs_mobilenet_v3_large

