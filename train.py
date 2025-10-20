from __future__ import print_function
import argparse
import sys
import time

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models.mitml.eval_metrics import eval_sysu, eval_regdb, evaluate
from models.simplebaseline_fourier_exchange.model_main import embed_net
from models.mitml.utils import *
# from models.simplebaseline_filip.loss import DualFineGrainedFrameDiffTripletLoss
from models.mitml.loss import OriTripletLoss
from models.simplebaseline_filip.loss import DualFineGrainedFrameDiffSepTripletLoss
from models.simplebaseline_filip.distance_metric import fine_grained_frame_distance, fine_grained_frame_distance_diff
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from data_manager import VCM, PRID2011, EventRgbDataManager
from data_loader import VideoDataset_train, VideoDataset_test, EventVideoDataset_test, FourierExchangeEventVideoDataset_train
# import transforms as T
from models.saadg.ChannelAug import ChannelRandomErasing

DATASET_MAPPING={
    'PRID2011': 'data/PRID2011',
    'VCM': 'data/VCM-HITSZ',
    'MARS': 'data/mars',
    'BUPT': 'data/BUPT'
}

print(torch.cuda.is_available())

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='PRID2011', help='dataset name: PRID2011(Video Cross-modal)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='network baseline:resnet34')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/mitml/event-rgb/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/mitml/', type=str,
                    help='log save path')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--vis_log_path', default='log/mitml/vis/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--seed', default=42, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
# for flip
parser.add_argument('--topk', default=1, type=int)
parser.add_argument('--test_avg', action='store_true', default=False)
parser.add_argument('--scale_distance', action='store_true', default=False)
parser.add_argument('--inference_no_norm', action='store_true', default=False)
parser.add_argument('--frame_bn', action='store_true', default=False)
parser.add_argument('--with_frame_id_loss', action='store_true', default=False)

parser.add_argument('--channel_erase', action='store_true', default=False)
parser.add_argument('--center_loss', action='store_true', default=False)
parser.add_argument('--fourier_lr_weight', type=float, default=0.01)

parser.add_argument('--return_raw_frame', action='store_true', default=False)
parser.add_argument('--with_global_tri', action='store_true', default=False)
parser.add_argument('--with_gem_pooling', action='store_true', default=False)

parser.add_argument('--grayscale_trans', action='store_true', default=False)
parser.add_argument('--sep_trans', action='store_true', default=False)
parser.add_argument('--sep_trans_mixup', action='store_true', default=False)
parser.add_argument('--window_ratio', type=float, default=1.0)
parser.add_argument('--mixup_alpha', type=float, default=1.0)
parser.add_argument('--sampling_mode', type=str, default='uniform')

parser.add_argument('--loss_kl_weight', type=float, default=0.)
parser.add_argument('--fid_loss_kl_weight', type=float, default=0.)

parser.add_argument('--fourier_track_l2_weight', type=float, default=0.)
parser.add_argument('--fourier_frame_l2_weight', type=float, default=0.)

parser.add_argument('--share_conv', action='store_true', default=False)
parser.add_argument('--share_bn', action='store_true', default=False)
parser.add_argument('--share_fc', action='store_true', default=False)

parser.add_argument('--global_loss_weight', type=float, default=0.)
parser.add_argument('--inter_modal_negative', action='store_true', default=False)
parser.add_argument('--metric_weight', type=float, default=1.)

parser.add_argument('--with_frame_id_loss_exchange', action='store_true', default=False)
parser.add_argument('--fourier_loss_weight', type=float, default=1.)

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(args.seed)
dataset = args.dataset

# 添加
seq_lenth = 6
test_batch = 32
#
data_set = EventRgbDataManager(DATASET_MAPPING[args.dataset])
log_path = args.log_path + args.dataset + '_CROSS_MODAL_EVENT_RGB_log/'

height = args.img_h
width = args.img_w

checkpoint_path = args.model_path
feature_dim_dic = {"resnet34":512, "resnet18":512, "resnet50":2048}
feature_dim = feature_dim_dic[args.arch]

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = dataset

suffix = suffix + '_{}_{}_{}_lr_{}_seed_{}'.format(args.exp_name, args.num_pos, args.batch_size, args.lr, args.seed)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_avg_map = -1

best_r1_e2v = 0  # best test accuracy
best_r1_v2e = 0
best_map_e2v = 0  # best test accuracy
best_map_v2e = 0

start_epoch = 0
wG = 0
end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train =[
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.Pad(10),

    transforms.RandomCrop((args.img_h, args.img_w)),
    # T.Random2DTranslation(height, width),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

if args.channel_erase:
    transform_train = transform_train + [ChannelRandomErasing(probability=0.5)]

transform_train = transforms.Compose(transform_train)

rgb_pos, event_pos = GenIdx(data_set.rgb_label, data_set.event_label)

test_event_loader = DataLoader(
    EventVideoDataset_test(data_set.test_event, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

test_rgb_loader = DataLoader(
    EventVideoDataset_test(data_set.test_rgb, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)


n_test_event = data_set.num_test_event_tracklets
n_test_rgb = data_set.num_test_rgb_tracklets

n_class = data_set.num_train_pids

print('==> Building model..')

net = embed_net(n_class, arch=args.arch, inference_norm=not args.inference_no_norm, frame_bn=args.frame_bn,
                share_conv=args.share_conv, share_bn=args.share_bn, share_fc=args.share_fc)
net.to(device)


if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion1 = nn.CrossEntropyLoss()
loader_batch = args.batch_size * args.num_pos
# criterion2 = DualFineGrainedFrameDiffTripletLoss(margin=args.margin)
criterion2 = DualFineGrainedFrameDiffSepTripletLoss(margin=args.margin, global_loss_weight=args.global_loss_weight,
                                                        inter_modal_negative=args.inter_modal_negative)

criterion3 = nn.KLDivLoss(reduction='batchmean')
criterion4 = nn.MSELoss()
criterion1.to(device)
criterion2.to(device)
criterion3.to(device)
criterion4.to(device)

# optimizer
# if args.optim == 'sgd':
#     ignored_params = list(map(id, net.bottleneck.parameters())) \
#                      + list(map(id, net.classifier.parameters()))
#
#     base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
#
#     optimizer_P = optim.SGD([
#         {'params': base_params, 'lr': 0.1 * args.lr},
#         {'params': net.bottleneck.parameters(), 'lr': args.lr},
#         {'params': net.classifier.parameters(), 'lr': args.lr},
#     ],
#         weight_decay=5e-4, momentum=0.9, nesterov=True)

# optimizer
if args.optim == 'sgd':

    ignored_params = list(map(id, net.base_resnet.parameters())) \
                     + list(map(id, net.visible_module.parameters())) \
                     + list(map(id, net.thermal_module.parameters()))

    base_params = filter(lambda p: id(p) in ignored_params, net.parameters())
    main_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': main_params, 'lr': args.lr}
    ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)


# def adjust_learning_rate(optimizer, epoch):
#     if epoch < 10:
#         lr = args.lr * (epoch + 1) / 10
#     elif 10 <= epoch < 35:
#         lr = args.lr
#     elif 35 <= epoch < 80:
#         lr = args.lr * 0.1
#     elif epoch >= 80:
#         lr = args.lr * 0.01
#
#     optimizer.param_groups[0]['lr'] = 0.1 * lr
#     for i in range(len(optimizer.param_groups) - 1):
#         optimizer.param_groups[i + 1]['lr'] = lr
#     return lr

def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 60:
        lr = args.lr
    elif 60 <= epoch < 120:
        lr = args.lr * 0.1
    elif epoch >= 120:
        lr = args.lr * 0.01

    # cur_lr = optimizer_P.param_groups[0]['lr']
    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    # optimizer_P.param_groups[0]['lr'] = lr

    return lr


def train(epoch, wG):
    # adjust learning rate
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss_original = AverageMeter()
    tri_loss_original = AverageMeter()
    id_loss_exchange = AverageMeter()
    tri_loss_exchange = AverageMeter()
    kl_loss = AverageMeter()
    fid_kl_loss = AverageMeter()
    fourier_track_l2_loss = AverageMeter()
    fourier_frame_l2_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (imgs_event, pids_event, camid_event, imgs_rgb, pids_rgb, camid_rgb, exchanged_images_rgb,
                    exchanged_images_event) in enumerate(trainloader):
        input1 = imgs_rgb
        input2 = imgs_event
        input3 = exchanged_images_rgb
        input4 = exchanged_images_event
        label1 = pids_rgb
        label2 = pids_event
        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        input3 = Variable(input3.cuda())
        input4 = Variable(input4.cuda())

        labels = Variable(labels.cuda())
        label1 = Variable(label1.cuda())
        label2 = Variable(label2.cuda())

        data_time.update(time.time() - end)

        feat_original, feat_exchanged, feat_frame_original, feat_frame_exchanged, logit_scale, out0, out1,\
            out2, out3 = net(input1, input2, exchanged_rgb=input3, exchanged_event=input4,
                             seq_len=seq_lenth, frame_id_loss=True, return_raw_frame=args.return_raw_frame,
                             gem_pooling=args.with_gem_pooling)

        loss_id_original = criterion1(out0, labels)
        loss_id_exchanged = criterion1(out1, labels)
        # KL loss
        loss_KL = 0.5 * (criterion3(F.log_softmax(out0, dim=1), F.softmax(out1, dim=1)) + \
                         criterion3(F.log_softmax(out1, dim=1), F.softmax(out0, dim=1)))
        # loss_KL = torch.tensor(0.).to(device)
        # fine-grained triplet loss
        if not args.scale_distance:
            logit_scale = None
        # loss_tri, batch_acc = criterion2(feat_frame_pool, labels, logit_scale, args.topk)
        
        # import pdb; pdb.set_trace()
        loss_tri_original, batch_acc = criterion2(feat_frame_original.clone(), labels, logit_scale, args.topk)
        loss_tri_exchanged, _ = criterion2(feat_frame_exchanged.clone(), labels, logit_scale, args.topk)

        # if args.center_loss:
        #     B, C = feat.shape
        #     feat = feat.reshape([B//2,2,C]).mean(dim=1)
        #     loss_center, _ = criterion2(feat, labels[0::2])
        #     loss_tri = loss_tri + loss_center
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        # add frame-level id loss
        local_label_1 = rearrange(repeat(label1.unsqueeze(dim=1), 'b 1 -> b n', n=seq_lenth), 'b n -> (b n)')
        local_label_2 = rearrange(repeat(label2.unsqueeze(dim=1), 'b 1 -> b n', n=seq_lenth), 'b n -> (b n)')
        local_label = torch.cat((local_label_1, local_label_2), dim=0)
        frame_id_loss_original = criterion1(out2, local_label)
        frame_id_loss_exchanged = criterion1(out3, local_label)

        frame_loss_KL = 0.5 * (criterion3(F.log_softmax(out2, dim=1), F.softmax(out3, dim=1)) + \
                         criterion3(F.log_softmax(out2, dim=1), F.softmax(out2, dim=1)))

        # L2 loss for feat w. and w/o fourier aug
        fourier_l2_track = criterion4(feat_original, feat_exchanged)
        fourier_l2_frame = criterion4(feat_frame_original, feat_frame_exchanged)

        loss = loss_id_original + loss_tri_original * args.metric_weight + args.fourier_loss_weight * (loss_id_exchanged +
               loss_tri_exchanged * args.metric_weight + args.loss_kl_weight * loss_KL + args.fid_loss_kl_weight * frame_loss_KL +
               args.fourier_track_l2_weight * fourier_l2_track + args.fourier_frame_l2_weight * fourier_l2_frame)

        if args.with_frame_id_loss:
            loss += frame_id_loss_original
        if args.with_frame_id_loss_exchange:
            loss += frame_id_loss_exchanged
        loss_total = loss
        # optimization
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('loss: ' + str(loss.cpu().detach().numpy()))
        # log different loss components
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss_original.update(loss_id_original.item(), 2 * input1.size(0))
        tri_loss_original.update(loss_tri_original.item(), 2 * input1.size(0))
        id_loss_exchange.update(loss_id_exchanged.item(), 2 * input1.size(0))
        tri_loss_exchange.update(loss_tri_exchanged.item(), 2 * input1.size(0))
        kl_loss.update(loss_KL.item(), 2 * input1.size(0))
        fid_kl_loss.update(frame_loss_KL.item(), 2 * input1.size(0))
        fourier_track_l2_loss.update(fourier_l2_track.item(), 2 * input1.size(0))
        fourier_frame_l2_loss.update(fourier_l2_frame.item(), 2 * input1.size(0))
        total += labels.size(0)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss_original: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss_original: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'iLoss_exchange: {id_loss_exchange.val:.4f} ({id_loss_exchange.avg:.4f}) '
                  'TLoss_exchange: {tri_loss_exchange.val:.4f} ({tri_loss_exchange.avg:.4f}) '
                  'KL loss: {kl_loss.val:.4f} ({kl_loss.avg:.4f}) '
                  'fid KL loss: {fid_kl_loss.val:.4f} ({fid_kl_loss.avg:.4f}) '
                  'fourier track l2 loss: {fourier_track_l2_loss.val:.4f} ({fourier_track_l2_loss.avg:.4f}) '
                  'fourier frame l2 loss: {fourier_frame_l2_loss.val:.4f} ({fourier_frame_l2_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss_original, tri_loss=tri_loss_original,
                id_loss_exchange=id_loss_exchange, tri_loss_exchange=tri_loss_exchange,
                kl_loss=kl_loss, fid_kl_loss=fid_kl_loss, fourier_track_l2_loss=fourier_track_l2_loss,
                fourier_frame_l2_loss=fourier_frame_l2_loss))
            # print(f'alpha: thermal: {net.thermal_module.alpha.item()} vis: {net.visible_module.alpha.item()}')
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss_original', id_loss_original.avg, epoch)
    writer.add_scalar('tri_loss_original', tri_loss_original.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)


def test2(epoch):
    """
    Evaluation for RGB2Event retrieval. The query set is rgb and the gallery set is the event.
    """
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((n_test_event, feature_dim))
    gall_frame_feat = np.zeros((n_test_event, seq_lenth, feature_dim))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(test_event_loader):
            input = imgs
            input = Variable(input.cuda())
            label = pids
            batch_num = input.size(0)
            feat, feat_frame_patch, feat_frame_pool = net(input, input, modal=2, seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_frame_feat[ptr:ptr + batch_num, :, :] = feat_frame_pool.detach().cpu().numpy()
            ptr = ptr + batch_num
            #
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((n_test_rgb, feature_dim))
    query_frame_feat = np.zeros((n_test_rgb, seq_lenth, feature_dim))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(test_rgb_loader):
            input = imgs
            label = pids

            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_frame_patch, feat_frame_pool = net(input, input, modal=1, seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_frame_feat[ptr:ptr + batch_num, :, :] = feat_frame_pool.detach().cpu().numpy()
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    # distmat = np.matmul(query_feat, np.transpose(gall_feat))
    # compute the fine-grained distance
    # distmat = fine_grained_frame_distance(torch.from_numpy(query_frame_feat), torch.from_numpy(gall_frame_feat))
    # distmat = distmat.detach().cpu().numpy()

    # compute the similarity
    if args.test_avg:
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
    else:
        # compute the fine-grained distance
        pass
        distmat = fine_grained_frame_distance_diff(torch.from_numpy(query_frame_feat), torch.from_numpy(gall_frame_feat),
                                                   None, args.topk, args.topk_smallest)
        distmat = -1 * distmat.detach().cpu().numpy()

    # evaluation
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP


def test(epoch):
    """
    Evaluation for Event2RGB retrieval (gallery set RGB, query set Event)
    """
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((n_test_rgb, feature_dim))
    gall_frame_feat = np.zeros((n_test_rgb, seq_lenth, feature_dim))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(test_rgb_loader):
            input = imgs
            label = pids
            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat, feat_frame_patch, feat_frame_pool = net(input, input, modal=1, seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_frame_feat[ptr:ptr + batch_num, :, :] = feat_frame_pool.detach().cpu().numpy()
            ptr = ptr + batch_num

            g_pids.extend(pids)
            g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((n_test_event, feature_dim))
    query_frame_feat = np.zeros((n_test_event, seq_lenth, feature_dim))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(test_event_loader):
            input = imgs
            label = pids

            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat, feat_frame_patch, feat_frame_pool = net(input, input, modal=2, seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_frame_feat[ptr:ptr + batch_num, :, :] = feat_frame_pool.detach().cpu().numpy()
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    # distmat = np.matmul(query_feat, np.transpose(gall_feat))
    # compute the fine-grained similarity
    # distmat = fine_grained_frame_distance(torch.from_numpy(query_frame_feat), torch.from_numpy(gall_frame_feat))
    # distmat = distmat.detach().cpu().numpy()

    # compute the similarity
    if args.test_avg:
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
    else:
        # compute the fine-grained distance
        pass
        distmat = fine_grained_frame_distance_diff(torch.from_numpy(query_frame_feat), torch.from_numpy(gall_frame_feat),
                                                   None, args.topk, args.topk_smallest)
        distmat = -1 * distmat.detach().cpu().numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc, mAP


# training
print('==> Start Training...')

for epoch in range(start_epoch, 200 - start_epoch):

    print('==> Preparing Data Loader...')
    sampler = IdentitySampler(data_set.event_label, data_set.rgb_label, rgb_pos, event_pos, args.num_pos, args.batch_size)
    index1 = sampler.index1
    index2 = sampler.index2
    loader_batch = args.batch_size * args.num_pos

    trainloader = DataLoader(
        FourierExchangeEventVideoDataset_train(data_set.train_event, data_set.train_rgb, seq_len=seq_lenth, sample='video_train',
                           transform=transform_train, index1=index1, index2=index2, grayscale_trans=args.grayscale_trans,
                           sep_trans=args.sep_trans, sep_trans_mixup=args.sep_trans_mixup, window_ratio=args.window_ratio,
                                               alpha=args.mixup_alpha, sampling_mode=args.sampling_mode),
        sampler=sampler,
        batch_size=loader_batch, num_workers=args.workers,
        drop_last=True,
    )

    # training
    wG = train(epoch, wG)

    if epoch >= 0 and (epoch+1) % 10 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc_e2v, mAP_e2v = test(epoch)

        print(
            'FC(e2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc_e2v[0], cmc_e2v[4], cmc_e2v[9], cmc_e2v[19], mAP_e2v))

        # print('Best e2v epoch [{}] r1 {:.2%} map {:.2%}'.format(best_epoch_e2v,best_acc,best_map_acc))
        print(
            'FC(e2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc_e2v[0], cmc_e2v[4], cmc_e2v[9], cmc_e2v[19], mAP_e2v), file=test_log_file)
        # -------------------------------------------------------------------------------------------------------------------
        cmc_v2e, mAP_v2e = test2(epoch)

        if (mAP_e2v + mAP_v2e)/2.0 > best_avg_map:
            best_avg_map = (mAP_e2v + mAP_v2e)/2.0
            best_map_v2e = mAP_v2e
            best_map_e2v = mAP_e2v
            best_epoch = epoch
            best_r1_v2e = cmc_v2e[0]
            best_r1_e2v = cmc_e2v[0]
            state = {
                'net': net.state_dict(),
                'mAP': best_avg_map,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print(
            'FC(v2e):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc_v2e[0], cmc_v2e[4], cmc_v2e[9], cmc_v2e[19], mAP_v2e))
        
        print(
            'FC(v2e):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc_v2e[0], cmc_v2e[4], cmc_v2e[9], cmc_v2e[19], mAP_v2e), file=test_log_file)
        print('Best epoch  [{}] v2e r1 {:.2%} map {:.2%} e2v r1 {:.2%} map {:.2%}'.format(best_epoch, best_r1_v2e, best_map_v2e, best_r1_e2v, best_map_e2v))
        test_log_file.flush()
