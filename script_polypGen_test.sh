#!/bin/bash

# -----> On test set
python polypGen_inference-seg.py --model deeplabv3plus_resnet101 --ckpt ./checkpoints_polypGen/best_deeplabv3plus_resnet101_voc_os16_polypGen.pth --gpu_id 0

python polypGen_inference-seg.py --model deeplabv3plus_resnet50 --ckpt ./checkpoints_polypGen/best_deeplabv3plus_resnet50_voc_os16_polypGen.pth --gpu_id 1

python polypGen_inference-seg.py --model pspNet --ckpt ./checkpoints_polypGen/best_pspNet_voc_os16_polypGen1.pth --gpu_id 2

python polypGen_inference-seg.py --model FCN8 --ckpt ./checkpoints_polypGen/best_FCN8_voc_os16_polypGen.pth --gpu_id 3


# ResNet-UNet models (ResNet34 and ResNet101)
python polypGen_inference-seg.py --model resnet-Unet --ckpt ./checkpoints_polypGen/best_resnet-Unet_voc_os16_polypGen_resnet34.pth --gpu_id 2 --backbone resnet34
# python polypGen_inference-seg.py --model resnet-Unet --ckpt ./checkpoints_polypGen1/best_resnet-Unet_voc_os16_polypGen1_resnet101.pth --gpu_id 3 --backbone resnet101

# 

# ------> Validation set 
# python polypGen_inference-seg_val.py --model deeplabv3plus_resnet101 --ckpt ./checkpoints_polypGen/best_deeplabv3plus_resnet101_voc_os16_polypGen.pth --gpu_id 1 
# python polypGen_inference-seg_val.py --model FCN8 --ckpt ./checkpoints_polypGen/best_FCN8_voc_os16_polypGen.pth --gpu_id 1 
# python polypGen_inference-seg_val.py --model pspNet --ckpt ./checkpoints_polypGen1/best_pspNet_voc_os16_polypGen.pth --gpu_id 1 
# python polypGen_inference-seg_val.py --model resnet-Unet --ckpt ./checkpoints_polypGen/best_resnet-Unet_voc_os16_polypGen_resnet101.pth --gpu_id 1 --backbone resnet101
# python polypGen_inference-seg_val.py --model resnet-Unet --ckpt ./checkpoints_polypGen/best_resnet-Unet_voc_os16_polypGen_resnet34.pth --gpu_id 1 --backbone resnet34