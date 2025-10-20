import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from .distance_metric import fine_grained_frame_distance, fine_grained_patch_distance, \
    fine_grained_frame_distance_diff, fine_grained_patch_distance_diff, masked_fine_grained_patch_distance_diff, \
    hausdorff_frame_distance_diff, hausdorff_frame_distance_cos, dual_hausdorff_frame_distance_diff, fine_grained_id_set_distance


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(), requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

# class MaskFineGrainedPatchDiffTripletLoss(nn.Module):
#     """
#     Fine-grained matching Triplet loss with hard positive/negative mining.
#     Using Difference instead of cosine similarity
#     Using hierarchical frame-patch matching
#
#     Modified from:
#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
#
#     Args:
#     - margin (float): margin for triplet.
#     """
#
#     def __init__(self, margin=0.3):
#         super(MaskFineGrainedPatchDiffTripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#
#     def forward(self, inputs, targets, logit_scale=None, topk=1, topk_smallest=False,
#                 patch_topk=1, raw_event=None, raw_rgb=None):
#         """
#         Args:
#         - inputs: feature matrix with shape (batch_size, seq_len, feat_dim, h, w)
#         - targets: ground truth labels with shape (num_classes)
#         """
#         n = inputs.size(0)
#
#         dist = fine_grained_patch_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
#                                                 topk_smallest=topk_smallest, patch_topk=patch_topk)
#
#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
#
#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#
#         # compute accuracy
#         correct = torch.ge(dist_an, dist_ap).sum().item()
#         return loss, correct


class MaskedFineGrainedPatchDiffTripletLoss(nn.Module):
    """
    Fine-grained matching Triplet loss with hard positive/negative mining.
    Using Difference instead of cosine similarity
    Using hierarchical frame-patch matching

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(MaskedFineGrainedPatchDiffTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, logit_scale=None, topk=1, topk_smallest=False, patch_topk=1,
                raw_rgb=None, raw_event=None):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim, h, w)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        dist = masked_fine_grained_patch_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                       topk_smallest=topk_smallest, patch_topk=patch_topk,
                                                       raw_rgb=raw_rgb, raw_event=raw_event)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class FineGrainedPatchDiffTripletLoss(nn.Module):
    """
    Fine-grained matching Triplet loss with hard positive/negative mining.
    Using Difference instead of cosine similarity
    Using hierarchical frame-patch matching

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(FineGrainedPatchDiffTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, logit_scale=None, topk=1, topk_smallest=False, patch_topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim, h, w)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        dist = fine_grained_patch_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk, topk_smallest=topk_smallest, patch_topk=patch_topk)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class FineGrainedFrameDiffTripletLoss(nn.Module):
    """
    Fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(FineGrainedFrameDiffTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, logit_scale=None, topk=1, topk_smallest=False):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        # print(f'input shape: {inputs.shape}')

        dist = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                topk_smallest=topk_smallest)
        # import pdb
        # pdb.set_trace()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class DualHausdorffDiffTripletLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, dist_metric='diff'):
        super(DualHausdorffDiffTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.dist_metric = dist_metric

    def forward(self, inputs, targets, logit_scale=None, topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        # print(f'input shape: {inputs.shape}')
        if self.dist_metric == 'diff':
            dist_max, dist_min = dual_hausdorff_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk)
        else:
            raise NotImplementedError
        # import pdb
        # pdb.set_trace()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_max[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_min[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



class HausdorffDiffTripletLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, dist_metric='diff'):
        super(HausdorffDiffTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.dist_metric = dist_metric

    def forward(self, inputs, targets, logit_scale=None, topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        # print(f'input shape: {inputs.shape}')
        if self.dist_metric == 'diff':
            dist = hausdorff_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk)
        elif self.dist_metric == 'cos':
            dist = hausdorff_frame_distance_cos(inputs, inputs, logit_scale=logit_scale, topk=topk)
        else:
            raise NotImplementedError
        # import pdb
        # pdb.set_trace()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class DualFineGrainedFrameDiffIdSetSepTripletLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, id_set_diagonal='both', global_loss_weight=0., id_set_weight=1.0,
                 id_set_diag_weight=1.0, id_topk=1, inter_modal_negative=False):
        super(DualFineGrainedFrameDiffIdSetSepTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.id_set_diagonal = id_set_diagonal
        self.id_set_weight = id_set_weight
        self.id_set_diag_weight = id_set_diag_weight
        self.global_loss_weight = global_loss_weight
        self.id_topk = id_topk
        self.inter_modal_negative = inter_modal_negative
    def forward(self, inputs, targets, logit_scale=None, topk=1, num_pos=2, id_topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        n_v = n // 2

        dist_largest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                        topk_smallest=False)
        dist_smallest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                         topk_smallest=True)
        dist_vv_largest, dist_vv_smallest = dist_largest[:n_v, :n_v], dist_smallest[:n_v, :n_v]
        dist_ee_largest, dist_ee_smallest = dist_largest[n_v:, n_v:], dist_smallest[n_v:, n_v:]
        dist_ve_largest, dist_ve_smallest = dist_largest[:n_v, n_v:], dist_smallest[:n_v, n_v:]
        dist_ev_largest, dist_ev_smallest = dist_largest[n_v:, :n_v], dist_smallest[n_v:, :n_v]

        # calculate the id set-based distance
        n_id = n // (2 * num_pos)
        id_dist_largest, _ = fine_grained_id_set_distance(dist_ve_largest, topk=self.id_topk)  # (n_id, n_id)
        _, id_dist_smallest = fine_grained_id_set_distance(dist_ve_smallest, topk=self.id_topk)  # (n_id, n_id)
        # For each anchor, find the hardest positive and negative
        id_target = targets[:n // 2].reshape(-1, num_pos)[:, 0]  # (n_id, )
        id_mask = id_target.expand(n_id, n_id).eq(id_target.expand(n_id, n_id).t())
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        id_dist_ap, id_dist_an = [], []
        for i in range(n_id):
            id_dist_ap.append(id_dist_largest[i][id_mask[i]].max().unsqueeze(0))
            id_dist_an.append(id_dist_smallest[i][id_mask[i] == 0].min().unsqueeze(0))
        for i in range(n_id):
            id_dist_ap.append(id_dist_largest.t()[i][id_mask[i]].max().unsqueeze(0))
            id_dist_an.append(id_dist_smallest.t()[i][id_mask[i] == 0].min().unsqueeze(0))
        id_dist_ap = torch.cat(id_dist_ap)
        id_dist_an = torch.cat(id_dist_an)
        id_y = torch.ones_like(id_dist_an)
        loss_id_set = self.ranking_loss(id_dist_an, id_dist_ap, id_y)
        loss_id_set_diag_largest = torch.sum(torch.diagonal(id_dist_largest))
        loss_id_set_diag_smallest = torch.sum(torch.diagonal(id_dist_smallest))
        if self.id_set_diagonal == 'both':
            loss_id_set_diag = loss_id_set_diag_smallest + loss_id_set_diag_largest
        elif self.id_set_diagonal == 'largest':
            loss_id_set_diag = loss_id_set_diag_largest
        elif self.id_set_diagonal == 'smallest':
            loss_id_set_diag = loss_id_set_diag_smallest
        else:
            raise ValueError

        targets_v = targets[:n_v]
        mask = targets_v.expand(n_v, n_v).eq(targets_v.expand(n_v, n_v).t())
        dist_apv, dist_ape, dist_apve, dist_apev, dist_anv, dist_ane, dist_anve, dist_anev = [], [], [], [], [], [], \
            [], []

        for i in range(n_v):
            # negative samples
            dist_anv.append(dist_vv_smallest[i][mask[i] == 0].min().unsqueeze(0))
            dist_ane.append(dist_ee_smallest[i][mask[i] == 0].min().unsqueeze(0))
            # inter-modal negative samples for compute acc only
            dist_anve.append(dist_ve_smallest[i][mask[i] == 0].min().unsqueeze(0))
            dist_anev.append(dist_ev_smallest[i][mask[i] == 0].min().unsqueeze(0))
            # intra-modal positive samples
            dist_apv.append(dist_vv_largest[i][mask[i]].max().unsqueeze(0))
            dist_ape.append(dist_ee_largest[i][mask[i]].max().unsqueeze(0))
            # inter-modal positive samples
            dist_apve.append(dist_ve_largest[i][mask[i]].max().unsqueeze(0))
            dist_apev.append(dist_ev_largest[i][mask[i]].max().unsqueeze(0))

        dist_apv = torch.cat(dist_apv)
        dist_ape = torch.cat(dist_ape)
        dist_apve = torch.cat(dist_apve)
        dist_apev = torch.cat(dist_apev)
        dist_anv = torch.cat(dist_anv)
        dist_ane = torch.cat(dist_ane)
        dist_anve = torch.cat(dist_anve)
        dist_anev = torch.cat(dist_anev)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_anv)
        loss_triple_v = self.ranking_loss(dist_anv, dist_apv, y) + self.ranking_loss(dist_anv, dist_apve, y)
        loss_triplet_e = self.ranking_loss(dist_ane, dist_ape, y) + self.ranking_loss(dist_ane, dist_apev, y)
        if self.inter_modal_negative:
            loss_triple_v += self.ranking_loss(dist_anve, dist_apv, y) + self.ranking_loss(dist_anve, dist_apve, y)
            loss_triplet_e += self.ranking_loss(dist_anev, dist_ape, y) + self.ranking_loss(dist_anev, dist_apev, y)
        # global center loss
        # calculate global loss
        cv = inputs[:n_v].mean(dim=1)
        ce = inputs[n_v:].mean(dim=1)
        cv_ = cv.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        ce_ = ce.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        loss_global = 0
        for i in range(n_v // num_pos):
            loss_global += (cv_[i] - ce_[i]).pow(2).sum().sqrt()
        loss = loss_triplet_e + loss_triple_v + self.global_loss_weight * loss_global
        loss += self.id_set_weight * loss_id_set + self.id_set_diag_weight * loss_id_set_diag
        # compute accuracy
        correct = torch.ge(dist_anve, dist_apve).sum().item() + torch.ge(dist_anev, dist_apev).sum().item()

        return loss, correct


class DualFineGrainedFrameDiffIdSetTripletLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(DualFineGrainedFrameDiffIdSetTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, logit_scale=None, topk=1, num_pos=2, id_topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        rgb_input, event_input = inputs[:n//2], inputs[n//2:]
        rgb_target, event_target = targets[:n//2], targets[n//2:]
        frame_dist_largest = fine_grained_frame_distance_diff(rgb_input, event_input, logit_scale=logit_scale, topk=topk,
                                                topk_smallest=False) #(n_id * num_pos, n_id * num_pos)
        frame_dist_smallest = fine_grained_frame_distance_diff(rgb_input, event_input, logit_scale=logit_scale, topk=topk,
                                                        topk_smallest=True)
        # calculate the id set-based distance
        n_id = n // (2 * num_pos)
        id_dist_largest, _ = fine_grained_id_set_distance(frame_dist_largest, topk=id_topk) #(n_id, n_id)
        _, id_dist_smallest = fine_grained_id_set_distance(frame_dist_smallest, topk=id_topk) #(n_id, n_id)

        # For each anchor, find the hardest positive and negative
        id_target = targets[:n//2].reshape(-1, num_pos)[:,0] #(n_id, )
        mask = id_target.expand(n_id, n_id).eq(id_target.expand(n_id, n_id).t())
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n_id):
            dist_ap.append(id_dist_largest[i][mask[i]].max().unsqueeze(0))
            dist_an.append(id_dist_smallest[i][mask[i] == 0].min().unsqueeze(0))
        for i in range(n_id):
            dist_ap.append(id_dist_largest.t()[i][mask[i]].max().unsqueeze(0))
            dist_an.append(id_dist_smallest.t()[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        loss_2 = torch.sum(torch.diagonal(id_dist_largest)) # enhance the positive distance

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss + loss_2, correct


class SepOriTripletLoss(nn.Module):
    def __init__(self, margin=0.3, global_loss_weight=0., inter_modal_negative=False):
        super(SepOriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.global_loss_weight = global_loss_weight
        self.inter_modal_negative = inter_modal_negative

    def forward(self, inputs, targets, logit_scale=None, topk=1, num_pos=2):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        n_v = n // 2

        # dist_largest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
        #                                         topk_smallest=False)
        # dist_smallest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
        #                                                 topk_smallest=True)

        track_feat = inputs.mean(dim=1)
        dist_1 = torch.pow(track_feat, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist_1 + dist_1.t()
        dist.addmm_(1, -2, track_feat, track_feat.t())
        dist = dist.clamp(min=1e-12).sqrt()
        # print(f'dist 4: {dist}')
        # exit()
        dist_smallest = dist
        dist_largest = dist
        # print(f'dist: {dist}')
        # exit()
        dist_vv_largest, dist_vv_smallest = dist_largest[:n_v, :n_v], dist_smallest[:n_v, :n_v]
        dist_ee_largest, dist_ee_smallest = dist_largest[n_v:, n_v:], dist_smallest[n_v:, n_v:]
        dist_ve_largest, dist_ve_smallest = dist_largest[:n_v, n_v:], dist_smallest[:n_v, n_v:]
        dist_ev_largest, dist_ev_smallest = dist_largest[n_v:, :n_v], dist_smallest[n_v:, :n_v]
        # print(f'dist vv: {dist_vv_largest}')
        # print(f'dist ee: {dist_ee_largest}')
        # print(f'dist ve: {dist_ve_largest}')
        # print(f'dist ev: {dist_ev_largest}')
        # exit()
        targets_v = targets[:n_v]
        mask = targets_v.expand(n_v, n_v).eq(targets_v.expand(n_v, n_v).t())
        # print(f'targets v: {targets_v}')
        # print(f'mask: {mask}')
        # exit()
        dist_apv, dist_ape, dist_apve, dist_apev, dist_anv, dist_ane, dist_anve, dist_anev = [], [], [], [], [], [], \
            [], []

        for i in range(n_v):
            # negative samples
            dist_anv.append(dist_vv_smallest[i][mask[i] == 0].min().unsqueeze(0))
            dist_ane.append(dist_ee_smallest[i][mask[i] == 0].min().unsqueeze(0))
            # inter-modal negative samples for compute acc only
            dist_anve.append(dist_ve_smallest[i][mask[i] == 0].min().unsqueeze(0))
            dist_anev.append(dist_ev_smallest[i][mask[i] == 0].min().unsqueeze(0))
            # intra-modal positive samples
            dist_apv.append(dist_vv_largest[i][mask[i]].max().unsqueeze(0))
            dist_ape.append(dist_ee_largest[i][mask[i]].max().unsqueeze(0))
            # inter-modal positive samples
            dist_apve.append(dist_ve_largest[i][mask[i]].max().unsqueeze(0))
            dist_apev.append(dist_ev_largest[i][mask[i]].max().unsqueeze(0))

        dist_apv = torch.cat(dist_apv)
        dist_ape = torch.cat(dist_ape)
        dist_apve = torch.cat(dist_apve)
        dist_apev = torch.cat(dist_apev)
        dist_anv = torch.cat(dist_anv)
        dist_ane = torch.cat(dist_ane)
        dist_anve = torch.cat(dist_anve)
        dist_anev = torch.cat(dist_anev)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_anv)
        loss_triple_v = self.ranking_loss(dist_anv, dist_apv, y) + self.ranking_loss(dist_anv, dist_apve, y)
        loss_triplet_e = self.ranking_loss(dist_ane, dist_ape, y) + self.ranking_loss(dist_ane, dist_apev, y)
        # print(f'loss v: {loss_triple_v}')
        # print(f'loss e: {loss_triplet_e}')
        if self.inter_modal_negative:
            loss_triple_v += self.ranking_loss(dist_anve, dist_apv, y) + self.ranking_loss(dist_anve, dist_apve, y)
            loss_triplet_e += self.ranking_loss(dist_anev, dist_ape, y) + self.ranking_loss(dist_anev, dist_apev, y)
        # global center loss
        # calculate global loss
        cv = track_feat[:n_v]
        ce = track_feat[n_v:]
        cv_ = cv.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        ce_ = ce.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        loss_global = 0
        for i in range(n_v // num_pos):
            loss_global += (cv_[i] - ce_[i]).pow(2).sum().sqrt()
        # print(f'loss gobal: {loss_global}')
        # exit()
        loss = loss_triplet_e + loss_triple_v + self.global_loss_weight * loss_global

        # compute accuracy
        correct = torch.ge(dist_anve, dist_apve).sum().item() + torch.ge(dist_anev, dist_apev).sum().item()
        return loss, correct


class MixedSepOriTripletLoss(nn.Module):
    def __init__(self, margin=0.3, global_loss_weight=0., inter_modal_negative=False):
        super(MixedSepOriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.global_loss_weight = global_loss_weight
        self.inter_modal_negative = inter_modal_negative

    def cal_dist_matrix(self, feat_a, feat_b):
        n = feat_a.size(0)
        track_feat_a = feat_a.mean(dim=1)
        track_feat_b = feat_b.mean(dim=1)
        dist_a = torch.pow(track_feat_a, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_b = torch.pow(track_feat_b, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist_a + dist_b.t()
        dist.addmm_(1, -2, track_feat_a, track_feat_b.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def cal_mix_sep_triplet_loss(self, dist, dist_mix, targets):
        n = dist.size(0)
        n_v = n // 2
        dist_vv_original = dist[:n_v, :n_v]
        dist_ee_original = dist[n_v:, n_v:]
        dist_vv_mix = dist_mix[:n_v, :n_v]
        dist_ve_mix = dist_mix[:n_v, n_v:]
        dist_ev_mix = dist_mix[n_v:, :n_v]
        dist_ee_mix = dist_mix[n_v:, n_v:]

        dist_apv, dist_ape, dist_apve, dist_apev, dist_anv, dist_ane = [], [], [], [], [], []
        targets_v = targets[:n_v]
        mask = targets_v.expand(n_v, n_v).eq(targets_v.expand(n_v, n_v).t())

        for i in range(n_v):
            # negative samples
            dist_anv.append(dist_vv_original[i][mask[i] == 0].min().unsqueeze(0))
            dist_ane.append(dist_ee_original[i][mask[i] == 0].min().unsqueeze(0))
            # intra-modal positive samples
            dist_apv.append(dist_vv_mix[i][mask[i]].max().unsqueeze(0))
            dist_ape.append(dist_ee_mix[i][mask[i]].max().unsqueeze(0))
            # inter-modal positive samples
            dist_apve.append(dist_ve_mix[i][mask[i]].max().unsqueeze(0))
            dist_apev.append(dist_ev_mix[i][mask[i]].max().unsqueeze(0))

        dist_apv = torch.cat(dist_apv)
        dist_ape = torch.cat(dist_ape)
        dist_apve = torch.cat(dist_apve)
        dist_apev = torch.cat(dist_apev)
        dist_anv = torch.cat(dist_anv)
        dist_ane = torch.cat(dist_ane)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_anv)
        loss_triple_v = self.ranking_loss(dist_anv, dist_apv, y) + self.ranking_loss(dist_anv, dist_apve, y)
        loss_triplet_e = self.ranking_loss(dist_ane, dist_ape, y) + self.ranking_loss(dist_ane, dist_apev, y)
        return loss_triplet_e + loss_triple_v


    def forward(self, inputs, inputs_exchanged, targets, logit_scale=None, topk=1, num_pos=2):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - inputs_exchanged: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        n_v = n // 2

        # dist_largest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
        #                                         topk_smallest=False)
        # dist_smallest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
        #                                                 topk_smallest=True)

        dist_original = self.cal_dist_matrix(inputs, inputs)
        dist_exchanged = self.cal_dist_matrix(inputs_exchanged, inputs_exchanged)
        dist_mix = self.cal_dist_matrix(inputs, inputs_exchanged)

        loss_tri_original = self.cal_mix_sep_triplet_loss(dist_original, dist_mix, targets)
        loss_tri_exchanged = self.cal_mix_sep_triplet_loss(dist_exchanged, dist_mix.t(), targets)

        # global center loss
        # calculate global loss
        cv_original = inputs[:n_v]
        ce_original = inputs[n_v:]
        cv_exchange = inputs_exchanged[:n_v]
        ce_exchange = inputs_exchanged[n_v:]
        cv_original_ = cv_original.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        ce_original_ = ce_original.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        cv_exchange_ = cv_exchange.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        ce_exchange_ = ce_exchange.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        loss_global = 0
        for i in range(n_v // num_pos):
            loss_global += (cv_original_[i] - cv_exchange_[i]).pow(2).sum().sqrt()
            loss_global += (ce_original_[i] - ce_exchange_[i]).pow(2).sum().sqrt()
        # print(f'loss gobal: {loss_global}')
        # exit()
        loss = loss_tri_original + loss_tri_exchanged + self.global_loss_weight * loss_global
        return loss


class DualFineGrainedFrameDiffSepTripletLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss.
    Use Difference instead of cosine similarity
    Separate for intra-modal and inter-modal positive samples.
    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, global_loss_weight=0., inter_modal_negative=False):
        super(DualFineGrainedFrameDiffSepTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.global_loss_weight = global_loss_weight
        self.inter_modal_negative = inter_modal_negative

    def forward(self, inputs, targets, logit_scale=None, topk=1, num_pos=2):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        n_v = n // 2

        dist_largest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                topk_smallest=False)
        dist_smallest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                        topk_smallest=True)
        dist_vv_largest, dist_vv_smallest = dist_largest[:n_v, :n_v], dist_smallest[:n_v, :n_v]
        dist_ee_largest, dist_ee_smallest = dist_largest[n_v:, n_v:], dist_smallest[n_v:, n_v:]
        dist_ve_largest, dist_ve_smallest = dist_largest[:n_v, n_v:], dist_smallest[:n_v, n_v:]
        dist_ev_largest, dist_ev_smallest = dist_largest[n_v:, :n_v], dist_smallest[n_v:, :n_v]
        targets_v = targets[:n_v]
        mask = targets_v.expand(n_v, n_v).eq(targets_v.expand(n_v, n_v).t())
        dist_apv, dist_ape, dist_apve, dist_apev, dist_anv, dist_ane, dist_anve, dist_anev = [], [], [], [], [], [], \
            [], []

        for i in range(n_v):
            # negative samples
            dist_anv.append(dist_vv_smallest[i][mask[i] == 0].min().unsqueeze(0))
            dist_ane.append(dist_ee_smallest[i][mask[i] == 0].min().unsqueeze(0))
            # inter-modal negative samples for compute acc only
            dist_anve.append(dist_ve_smallest[i][mask[i] == 0].min().unsqueeze(0))
            dist_anev.append(dist_ev_smallest[i][mask[i] == 0].min().unsqueeze(0))
            # intra-modal positive samples
            dist_apv.append(dist_vv_largest[i][mask[i]].max().unsqueeze(0))
            dist_ape.append(dist_ee_largest[i][mask[i]].max().unsqueeze(0))
            # inter-modal positive samples
            dist_apve.append(dist_ve_largest[i][mask[i]].max().unsqueeze(0))
            dist_apev.append(dist_ev_largest[i][mask[i]].max().unsqueeze(0))

        dist_apv = torch.cat(dist_apv)
        dist_ape = torch.cat(dist_ape)
        dist_apve = torch.cat(dist_apve)
        dist_apev = torch.cat(dist_apev)
        dist_anv = torch.cat(dist_anv)
        dist_ane = torch.cat(dist_ane)
        dist_anve = torch.cat(dist_anve)
        dist_anev = torch.cat(dist_anev)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_anv)
        loss_triple_v = self.ranking_loss(dist_anv, dist_apv, y) + self.ranking_loss(dist_anv, dist_apve, y)
        loss_triplet_e = self.ranking_loss(dist_ane, dist_ape, y) + self.ranking_loss(dist_ane, dist_apev, y)
        if self.inter_modal_negative:
            loss_triple_v += self.ranking_loss(dist_anve, dist_apv, y) + self.ranking_loss(dist_anve, dist_apve, y)
            loss_triplet_e += self.ranking_loss(dist_anev, dist_ape, y) + self.ranking_loss(dist_anev, dist_apev, y)
        # global center loss
        # calculate global loss
        cv = inputs[:n_v].mean(dim=1)
        ce = inputs[n_v:].mean(dim=1)
        cv_ = cv.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        ce_ = ce.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        loss_global = 0
        for i in range(n_v // num_pos):
            loss_global += (cv_[i] - ce_[i]).pow(2).sum().sqrt()
        loss = loss_triplet_e + loss_triple_v + self.global_loss_weight * loss_global

        # compute accuracy
        correct = torch.ge(dist_anve, dist_apve).sum().item() + torch.ge(dist_anev, dist_apev).sum().item()
        return loss, correct


class DualFineGrainedFrameDiffTripletGlobalLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, global_loss_weight=1.0):
        super(DualFineGrainedFrameDiffTripletGlobalLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.global_loss_weight = global_loss_weight

    def forward(self, inputs, targets, logit_scale=None, topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        # print(f'input shape: {inputs.shape}')

        dist_largest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                topk_smallest=False)
        dist_smallest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                        topk_smallest=True)
        # import pdb
        # pdb.set_trace()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_largest[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_smallest[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, dist_ap, y)

        # Compute global loss
        # calculate global loss
        n_v = n // 2
        cv = inputs[:n_v].mean(dim=1)
        ce = inputs[n_v:].mean(dim=1)
        cv_ = cv.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        ce_ = ce.reshape(n_v // num_pos, num_pos, -1).mean(dim=1)
        loss_global = 0
        for i in range(n_v // num_pos):
            loss_global += (cv_[i] - ce_[i]).pow(2).sum().sqrt()
        loss = loss_triplet + self.global_loss_weight * loss_global

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class DualFineGrainedFrameDiffTripletLoss(nn.Module):
    """
    Dual fine-grained matching Triplet loss with hard positive/negative mining.
    Use Difference instead of cosine similarity

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(DualFineGrainedFrameDiffTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, logit_scale=None, topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        # print(f'input shape: {inputs.shape}')

        dist_largest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                topk_smallest=False)
        dist_smallest = fine_grained_frame_distance_diff(inputs, inputs, logit_scale=logit_scale, topk=topk,
                                                        topk_smallest=True)
        # import pdb
        # pdb.set_trace()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_largest[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_smallest[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class FineGrainedFrameTripletLoss(nn.Module):
    """
    Fine-grained matching Triplet loss with hard positive/negative mining.

    Modified from:
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(FineGrainedFrameTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, logit_scale=None, topk=1):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, seq_len, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # # compute the distance using fine-grained frame distance function
        # print(f'input shape: {inputs.shape}')

        dist = fine_grained_frame_distance(inputs, inputs, logit_scale=logit_scale, topk=topk)
        if logit_scale is not None:
            dist = logit_scale.detach().cpu().item() - dist
        else:
            dist = 1.0 - dist
        # import pdb
        # pdb.set_trace()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, batch_size, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct*2
        
class BiTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BiTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, input, target):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = self.batch_size
        input1 = input.narrow(0,0,n)
        input2 = input.narrow(0,n,n)
        
        # Compute pairwise distance, replace by the official when merged
        dist = pdist_torch(input1, input2)
        
        # For each anchor, find the hardest positive and negative
        # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i,i].unsqueeze(0))
            dist_an.append(dist[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss1 = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct1  =  torch.ge(dist_an, dist_ap).sum().item() 
        
        # Compute pairwise distance, replace by the official when merged
        dist2 = pdist_torch(input2, input1)
        
        # For each anchor, find the hardest positive and negative
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist2[i,i].unsqueeze(0))
            dist_an2.append(dist2[i][self.mask[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        
        # Compute ranking hinge loss
        y2 = torch.ones_like(dist_an2)
        # loss2 = self.ranking_loss(dist_an2, dist_ap2, y2)
        
        loss2 = torch.sum(torch.nn.functional.relu(dist_ap2 + self.margin - dist_an2))
        
        # compute accuracy
        correct2  =  torch.ge(dist_an2, dist_ap2).sum().item()
        
        loss = torch.add(loss1, loss2)
        return loss, correct1 + correct2
        
        
class BDTRLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.suffix
    """
    def __init__(self, batch_size, margin=0.5):
        super(BDTRLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.batch_size = batch_size
        self.mask = torch.eye(batch_size)
    def forward(self, inputs, targets):
        """
        Args:
        - input: feature matrix with shape (batch_size, feat_dim)
        - target: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct  =  torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx