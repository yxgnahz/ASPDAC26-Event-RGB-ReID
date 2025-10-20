#!/bin/bash

python train.py --dataset VCM --lr 0.05 --batch-size 4 --topk 2 --test_avg \
--frame_bn --margin 0.3 --arch resnet50 --channel_erase --return_raw_frame --sep_trans_mixup --window_ratio 1.0 \
--sampling_mode uniform --mixup_alpha 1.0 --with_frame_id_loss --with_frame_id_loss_exchange --loss_kl_weight 0.1 \
--global_loss_weight 1.0 --fourier_loss_weight 0.5
