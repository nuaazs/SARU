#!/bin/bash
# GPU 0
# ResNet + Pix2Pix + ResNet_Attn + selfattentionresunet

# GPU 1
# SARU + SARU_f + UNet_128 + UNet_128_Attn + cbamunet

conda activate antsnetpy
#python train.py --netG resnet --name ResNet_0630 --gpu 0 > ./log/ResNet_0630.log && \
#python train.py --netG unet_128 --name Pix2Pix_0630 --gpu 0 --model pix2pix > ./log/Pix2Pix_0630.log && \
python train.py --netG resnet_attn --name ResNet_Attn_0630 --gpu_ids 0 > ./log/ResNet_Attn_0630.log  && \
python train.py --netG selfattentionresunet --name SARU_v2_0630 --gpu_ids 0 > ./log/SARU_v2_0630.log

# python train.py --netG saru --name SARU_v1_0630 --gpu 1 > ./log/SARU_v1_0630.log && \
# python train.py --netG saru --name SARUf_v1_0630 --gpu 1 --dataroot /mnt/zhaosheng/mrct/data/mrctf > ./log/SARUf_v1_0630.log && \
# python train.py --netG unet_128 --name UNet_128_0630 --gpu 1 > ./log/UNet_128_0630.log && \
# python train.py --netG unet_128_attn --name UNet_128_Attn_0630 --gpu 1 > ./log/UNet_128_Attn_0630.log && \
# python train.py --netG cbamunet --name CBAM_UNet_0630 --gpu 1 > ./log/CBAM_UNet_0630.log


