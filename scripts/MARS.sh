#!/bin/bash

#SBATCH --job-name=event
#SBATCH --mail-user=xyzhang21@link.cuhk.edu.hk
#SBATCH --output=./log/flip-fourier-exchange/MARS_bs4_res50_lr005_slce_fexchange05_filip_sep_tri_top2_global10_sep_mixup_uniform10_window10_kl01_bn_m03_rtraw_fidloss.log
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=3090
#SBATCH --exclude=proj[77,73,193]
python train.py --dataset MARS --lr 0.05 --batch-size 4 --topk 2 --test_avg \
--frame_bn --margin 0.3 --arch resnet50 --channel_erase --return_raw_frame --sep_trans_mixup --window_ratio 1.0 \
--sampling_mode uniform --mixup_alpha 1.0 --with_frame_id_loss --with_frame_id_loss_exchange --loss_kl_weight 0.1 \
--global_loss_weight 1.0 --fourier_loss_weight 0.5
