import torch


def fine_grained_frame_distance(feat_1, feat_2, logit_scale=None, topk=1):
    """
    input:
        feat_1 shape (N1, seq_len, hid_dim)
        feat_2 shape (N2, seq_len, hid_dim)
    return:
        distance (N1, N2)
    """
    b1, t1, hid_dim1 = feat_1.shape
    b2, t2, hid_dim2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    feat_1 = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    feat_2 = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k = f1.shape
        l, m, k = f2.shape
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        logits_1 = scale * dense_feat_1 @ dense_feat_2.t()
        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1

    final_logits_1 = get_logits(feat_1, feat_2).topk(topk, dim=-1)[0].mean(dim=(2,3)) #(b1, b2)
    final_logits_2 = get_logits(feat_2, feat_1).topk(topk, dim=-1)[0].mean(dim=(2,3)) #(b2, b1)

    final_logits = 0.5 * (final_logits_1 + final_logits_2.t()) # (b1, b2)

    return final_logits


def dual_hausdorff_frame_distance_diff(feat_1, feat_2, logit_scale=None, topk=1):
    """
    use the difference instead of cosine similarity
    input:
        feat_1 shape (N1, seq_len, hid_dim)
        feat_2 shape (N2, seq_len, hid_dim)
    return:
        distance (N1, N2)
    """
    b1, t1, hid_dim1 = feat_1.shape
    b2, t2, hid_dim2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    # feat_1 = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    # feat_2 = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k = f1.shape
        l, m, k = f2.shape
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        # Compute pairwise distance
        dist_1 = torch.pow(dense_feat_1, 2).sum(dim=1, keepdim=True).expand(i*j, l*m)
        dist_2 = torch.pow(dense_feat_2, 2).sum(dim=1, keepdim=True).expand(l*m, i*j)
        dist = dist_1 + dist_2.t()
        dist.addmm_(1, -2, dense_feat_1, dense_feat_2.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        logits_1 = scale * dist
        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1

    final_logits = get_logits(feat_1, feat_2) # (b1, b2, t1, t2)

    hausdorff_dist_1 = final_logits.topk(1, dim=-1, largest=False)[0].topk(topk, dim=-2,
                                                                           largest=True)[0][:, :, -1].squeeze()
    hausdorff_dist_2 = final_logits.topk(1, dim=-2, largest=False)[0].topk(topk, dim=-1,
                                                                           largest=True)[0][:, :, :, -1].squeeze()
    final_logits_max = torch.maximum(hausdorff_dist_1, hausdorff_dist_2)
    final_logits_min = torch.minimum(hausdorff_dist_1, hausdorff_dist_2)

    return final_logits_max, final_logits_min


def hausdorff_frame_distance_diff(feat_1, feat_2, logit_scale=None, topk=1):
    """
    use the difference instead of cosine similarity
    input:
        feat_1 shape (N1, seq_len, hid_dim)
        feat_2 shape (N2, seq_len, hid_dim)
    return:
        distance (N1, N2)
    """
    b1, t1, hid_dim1 = feat_1.shape
    b2, t2, hid_dim2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    # feat_1 = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    # feat_2 = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k = f1.shape
        l, m, k = f2.shape
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        # Compute pairwise distance
        dist_1 = torch.pow(dense_feat_1, 2).sum(dim=1, keepdim=True).expand(i*j, l*m)
        dist_2 = torch.pow(dense_feat_2, 2).sum(dim=1, keepdim=True).expand(l*m, i*j)
        dist = dist_1 + dist_2.t()
        dist.addmm_(1, -2, dense_feat_1, dense_feat_2.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        logits_1 = scale * dist
        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1

    final_logits = get_logits(feat_1, feat_2) # (b1, b2, t1, t2)

    hausdorff_dist_1 = final_logits.topk(1, dim=-1, largest=False)[0].topk(topk, dim=-2,
                                                                           largest=True)[0][:, :, -1].squeeze()
    hausdorff_dist_2 = final_logits.topk(1, dim=-2, largest=False)[0].topk(topk, dim=-1,
                                                                           largest=True)[0][:, :, :, -1].squeeze()
    final_logits = torch.maximum(hausdorff_dist_1, hausdorff_dist_2)

    return final_logits


def hausdorff_frame_distance_cos(feat_1, feat_2, logit_scale=None, topk=1):
    """
    use the difference instead of cosine similarity
    input:
        feat_1 shape (N1, seq_len, hid_dim)
        feat_2 shape (N2, seq_len, hid_dim)
    return:
        distance (N1, N2)
    """
    b1, t1, hid_dim1 = feat_1.shape
    b2, t2, hid_dim2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    feat_1 = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    feat_2 = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k = f1.shape
        l, m, k = f2.shape
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        # Compute pairwise distance

        dist = dense_feat_1 @ dense_feat_2.t()

        logits_1 = scale * dist
        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1

    final_logits = get_logits(feat_1, feat_2) # (b1, b2, t1, t2)

    hausdorff_dist_1 = final_logits.topk(1, dim=-1, largest=False)[0].topk(topk, dim=-2,
                                                                           largest=True)[0][:, :, -1].squeeze()
    hausdorff_dist_2 = final_logits.topk(1, dim=-2, largest=False)[0].topk(topk, dim=-1,
                                                                           largest=True)[0][:, :, :, -1].squeeze()
    final_logits = torch.maximum(hausdorff_dist_1, hausdorff_dist_2)

    return final_logits


def fine_grained_frame_distance_diff(feat_1, feat_2, logit_scale=None, topk=1, topk_smallest=False):
    """
    use the difference instead of cosine similarity
    input:
        feat_1 shape (N1, seq_len, hid_dim)
        feat_2 shape (N2, seq_len, hid_dim)
    return:
        distance (N1, N2)
    """
    b1, t1, hid_dim1 = feat_1.shape
    b2, t2, hid_dim2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    # feat_1 = feat_1 / feat_1.norm(dim=-1, keepdim=True)
    # feat_2 = feat_2 / feat_2.norm(dim=-1, keepdim=True)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k = f1.shape
        l, m, k = f2.shape
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        # Compute pairwise distance
        dist_1 = torch.pow(dense_feat_1, 2).sum(dim=1, keepdim=True).expand(i*j, l*m)
        dist_2 = torch.pow(dense_feat_2, 2).sum(dim=1, keepdim=True).expand(l*m, i*j)
        dist = dist_1 + dist_2.t()
        dist.addmm_(1, -2, dense_feat_1, dense_feat_2.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        logits_1 = scale * dist
        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1

    final_logits_1 = get_logits(feat_1, feat_2).topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3)) #(b1, b2)
    final_logits_2 = get_logits(feat_2, feat_1).topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3)) #(b2, b1)

    final_logits = 0.5 * (final_logits_1 + final_logits_2.t()) # (b1, b2)

    return final_logits


def fine_grained_patch_distance_diff(feat_1, feat_2, logit_scale=None, topk=1, topk_smallest=False, patch_topk = 1):
    """
    feat_1, feat_2 of shape (N, seq_len, hid_dim, h, w)
    """
    b1, t1, hid_dim1, h1, w1 = feat_1.shape
    b2, t2, hid_dim2, h2, w2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    assert h1 == h2
    assert w1 == w2

    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k, h1, w1 = f1.shape
        l, m, k, h2, w2 = f2.shape
        f1 = f1.permute(0, 1, 3, 4, 2)
        f2 = f2.permute(0, 1, 3, 4, 2)
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        patch_distance_1 = get_patch_distance(dense_feat_1, dense_feat_2) # (ijh1w1, lmh2w2)
        patch_distance_2 = get_patch_distance(dense_feat_2, dense_feat_1) # (lmh2w2, ijh1w1)

        patch_distance_1 = patch_distance_1.reshape(i, j, h1*w1, l, m, h2*w2).topk(patch_topk, dim=-1, largest=False)[0].mean(dim=(2,5)) # (i,j,l,m)
        patch_distance_2 = patch_distance_2.reshape(l, m, h2*w2, i, j, h1*w1).topk(patch_topk, dim=-1, largest=False)[0].mean(dim=(2,5)).permute(2,3,0,1) # (l,m,i,j)
        logits_1 = 0.5 * scale * (patch_distance_1 + patch_distance_2)

        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1

    def get_patch_distance(f1, f2):
        i, k = f1.shape
        j, k = f2.shape
        dist_1 = torch.pow(f1, 2).sum(dim=1, keepdim=True).expand(i, j)
        dist_2 = torch.pow(f2, 2).sum(dim=1, keepdim=True).expand(j, i)
        dist = dist_1 + dist_2.t()
        dist.addmm_(1, -2, f1, f2.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


    final_logits_1 = get_logits(feat_1, feat_2).topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3)) #(b1, b2)
    final_logits_2 = get_logits(feat_2, feat_1).topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3)) #(b2, b1)

    final_logits = 0.5 * (final_logits_1 + final_logits_2.t()) # (b1, b2)

    return final_logits


def masked_patch_distance_diff(feat_1, feat_2, mask_1=None, mask_2=None, topk=1, topk_smallest=False, patch_topk=1):
    """
    feat_1, feat_2 of shape (N, seq_len, hid_dim, h, w)
    """
    b1, t1, hid_dim1, h1, w1 = feat_1.shape
    b2, t2, hid_dim2, h2, w2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    assert h1 == h2
    assert w1 == w2

    def get_frame_logits(f1, f2):
        """
        Input:
            f1: (N1, T1, C, M1)
            f2: (N2, T2, C, M2)
        Output:
            Patch-wise similarity: (N1, N2, T1, T2)
        """
        n1, t1, c, m1 = f1.shape
        n2, t2, c, m2 = f2.shape
        f1 = f1.permute(0, 1, 3, 2)
        f2 = f2.permute(0, 1, 3, 2)
        dense_feat_1 = f1.reshape(-1, c)
        dense_feat_2 = f2.reshape(-1, c)
        patch_distance = get_patch_distance(dense_feat_1, dense_feat_2)  # (n1t1m1, n2t2m2)

        patch_distance_1 = \
            patch_distance.reshape(n1, t1, m1, n2, t2, m2).topk(patch_topk,
                                    dim=-1, largest=False)[0].mean(dim=(2, 5))  # (n1,t1,n2,t2)
        patch_distance_2 = \
            patch_distance.reshape(n1, t1, m1, n2, t2, m2).permute(3, 4, 5, 0, 1, 2).topk(patch_topk,
                                   dim=-1, largest=False)[0].mean(dim=(2, 5)).permute(2, 3, 0, 1)  # (n1,t1,n2,t2)
        logits = 0.5 * (patch_distance_1 + patch_distance_2)

        logits = logits.permute(0, 2, 1, 3)  # (n1, n2, t1, t2)
        return logits

    def get_patch_distance(f1, f2):
        i, k = f1.shape
        j, k = f2.shape
        dist_1 = torch.pow(f1, 2).sum(dim=1, keepdim=True).expand(i, j)
        dist_2 = torch.pow(f2, 2).sum(dim=1, keepdim=True).expand(j, i)
        dist = dist_1 + dist_2.t()
        dist.addmm_(1, -2, f1, f2.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    if mask_1 is not None:
        feat_1 = mask_event_feature(mask_1, feat_1)
    else:
        feat_1 = feat_1.reshape(b1, t1, hid_dim1, -1)
    if mask_2 is not None:
        feat_2 = mask_event_feature(mask_2, feat_2)
    else:
        feat_2 = feat_2.reshape(b2, t2, hid_dim2, -1)

    raw_frame_logits = get_frame_logits(feat_1, feat_2)

    frame_logits = 0.5 * (raw_frame_logits.topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3))+
                     raw_frame_logits.topk(topk, dim=-2, largest=not topk_smallest)[0].mean(dim=(2, 3)))

    return frame_logits


def masked_fine_grained_patch_distance_diff(feat_1, feat_2, logit_scale=None, topk=1, topk_smallest=False,
                                            raw_event=None, raw_rgb=None, patch_topk=1):
    """
    feat_1, feat_2 of shape (N, seq_len, hid_dim, h, w)
    """
    b1, t1, hid_dim1, h1, w1 = feat_1.shape
    b2, t2, hid_dim2, h2, w2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    assert h1 == h2
    assert w1 == w2

    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_frame_logits(f1, f2):
        """
        Input:
            f1: (N1, T1, C, M1)
            f2: (N2, T2, C, M2)
        Output:
            Patch-wise similarity: (N1, N2, T1, T2)
        """
        n1, t1, c, m1 = f1.shape
        n2, t2, c, m2 = f2.shape
        f1 = f1.permute(0, 1, 3, 2)
        f2 = f2.permute(0, 1, 3, 2)
        dense_feat_1 = f1.reshape(-1, c)
        dense_feat_2 = f2.reshape(-1, c)
        patch_distance = get_patch_distance(dense_feat_1, dense_feat_2)  # (n1t1m1, n2t2m2)

        patch_distance_1 = \
            patch_distance.reshape(n1, t1, m1, n2, t2, m2).topk(patch_topk,
                                    dim=-1, largest=False)[0].mean(dim=(2, 5))  # (n1,t1,n2,t2)
        patch_distance_2 = \
            patch_distance.reshape(n1, t1, m1, n2, t2, m2).permute(3, 4, 5, 0, 1, 2).topk(patch_topk,
                                   dim=-1, largest=False)[0].mean(dim=(2, 5)).permute(2, 3, 0, 1)  # (n1,t1,n2,t2)
        logits = 0.5 * scale * (patch_distance_1 + patch_distance_2)

        logits = logits.permute(0, 2, 1, 3)  # (n1, n2, t1, t2)
        return logits

    def get_patch_distance(f1, f2):
        i, k = f1.shape
        j, k = f2.shape
        dist_1 = torch.pow(f1, 2).sum(dim=1, keepdim=True).expand(i, j)
        dist_2 = torch.pow(f2, 2).sum(dim=1, keepdim=True).expand(j, i)
        dist = dist_1 + dist_2.t()
        dist.addmm_(1, -2, f1, f2.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    event_mask = get_event_mask(raw_event, (h1, w1), 1) # the two event features share the same masks
    masked_event_features_1 = mask_event_feature(event_mask, feat_1[b1//2:, :])
    #print(f'mask event featurs 1 shape: {masked_event_features_1.shape}')
    masked_event_features_2 = mask_event_feature(event_mask, feat_2[b2//2:, :])
    #print(f'rgb raw features shape: {feat_1.shape}')
    rgb_features_1 = feat_1[:b1//2, :].reshape(b1//2, t1, hid_dim1, -1)
    #print(f'rgb features 1 shape: {rgb_features_1.shape}')
    rgb_features_2 = feat_2[:b2//2, :].reshape(b2//2, t2, hid_dim2, -1)

    rgb1_to_rgb2_frame_logits = get_frame_logits(rgb_features_1, rgb_features_2)
    #print(f'rgb2rgb frame logits: {rgb1_to_rgb2_frame_logits.shape}')
    rgb1_to_event2_frame_logits = get_frame_logits(rgb_features_1, masked_event_features_2)
    #print(f'rgb2 event frame logits: {rgb1_to_event2_frame_logits.shape}')
    event1_to_rgb2_frame_logits = get_frame_logits(masked_event_features_1, rgb_features_2)
    #print(f'event2rgb frame logits: {event1_to_rgb2_frame_logits.shape}')
    event1_to_event2_frame_logits = get_frame_logits(masked_event_features_1, masked_event_features_2)
    #print(f'event2event frame logits: {event1_to_event2_frame_logits.shape}')

    rgb2rgb_logits = 0.5 * (rgb1_to_rgb2_frame_logits.topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3))+
                     rgb1_to_rgb2_frame_logits.topk(topk, dim=-2, largest=not topk_smallest)[0].mean(dim=(2, 3)))
    #print(f'rgb2rgb logits: {rgb2rgb_logits.shape}')
    rgb2event_logits = 0.5 * (rgb1_to_event2_frame_logits.topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3))+
                       rgb1_to_event2_frame_logits.topk(topk, dim=-2, largest=not topk_smallest)[0].mean(dim=(2, 3)))
    #print(f'rgb2event logits: {rgb2event_logits.shape}')
    event2rgb_logits = 0.5 * (event1_to_rgb2_frame_logits.topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3))+
                       event1_to_rgb2_frame_logits.topk(topk, dim=-2, largest=not topk_smallest)[0].mean(dim=(2, 3)))
    #print(f'event2rgb logits: {event2rgb_logits.shape}')
    event2event_logits = 0.5 * (event1_to_event2_frame_logits.topk(topk, dim=-1, largest=not topk_smallest)[0].mean(dim=(2, 3))+
                         event1_to_event2_frame_logits.topk(topk, dim=-2, largest=not topk_smallest)[0].mean(dim=(2, 3)))
    #print(f'event2event logits: {event2event_logits.shape}')

    final_logits_1 = torch.cat((rgb2rgb_logits, rgb2event_logits), 1)
    #print(f'final logits1 shape: {final_logits_1.shape}')
    final_logits_2 = torch.cat((event2rgb_logits, event2event_logits), 1)
    #print(f'final logits2 shape: {final_logits_2.shape}')
    final_logits = torch.cat((final_logits_1, final_logits_2), 0)
    #print(f'final logits shape: {final_logits.shape}')
    return final_logits


def mask_event_feature(event_mask, event_features, ratio=0.2):
    """
    Input:
        event_mask: (N, T, H, W)
        event_features: (N, T, C, H, W)
    Return:
        masked_event_features: (N, T, C, M) where M is the number of features selected
    """
    N, T, C, H, W = event_features.shape
    event_mask = event_mask.view(N*T, -1)
    # print(f'event mask: {event_mask[0]}')
    num_selected_values = int(ratio * event_mask.shape[1])
    _, indices = torch.topk(event_mask, num_selected_values, dim=1) # (NT, M)
    # print(f'indices: {indices[0]}')
    # exit()
    event_features = event_features.permute(0, 1, 3, 4, 2) #(N, T, H, W, C)
    event_features = event_features.reshape(N*T, H*W, C)
    masked_event_features = torch.gather(event_features, 1, indices.unsqueeze(-1).expand(-1, -1, C).to(event_features.device)) # (NT, M, C)
    masked_event_features = masked_event_features.permute(0, 2, 1).reshape(N, T, C, -1)
    return masked_event_features


def get_event_mask(raw_event_img, feature_shape, visualization=False):
    """
    Extract the patches with rich event information.
    input:
        raw_event_img: (N, T, C, H, W)
        feature_shape: (H_feat, W_feat)
    return:
        patch index
    """
    N, T, C, H, W = raw_event_img.shape
    raw_event_pos, raw_event_neg = raw_event_img[:, :, 0, :, :], raw_event_img[:, :, 1, :, :]
    raw_event_pos = raw_event_pos.reshape(-1, H, W)
    raw_event_neg = raw_event_neg.reshape(-1, H, W)

    def calculate_non_zero_elements(raw_event_frame, feature_shape):
        _, H, W = raw_event_frame.shape
        H_feat, W_feat = feature_shape
        all_non_zero_counts = list()
        for f in raw_event_frame:
            non_zero_indices = torch.nonzero(f, as_tuple=True)
            non_zero_counts = torch.zeros(feature_shape, dtype=torch.int)
            rows, cols = [], []
            for i in range(non_zero_indices[0].shape[0]):
                row, col = (non_zero_indices[0][i] * H_feat) // H, (non_zero_indices[1][i] * W_feat) // W
                rows.append(row)
                cols.append(col)
                non_zero_counts[row, col] += 1
            all_non_zero_counts.append(non_zero_counts)
        return torch.stack(all_non_zero_counts)

    event_neg_mask = calculate_non_zero_elements(raw_event_neg, feature_shape)  # (N, H_feat, W_feat)
    event_pos_mask = calculate_non_zero_elements(raw_event_pos, feature_shape) # (N, H_feat, W_feat)
    event_mask = event_pos_mask + event_neg_mask

    if visualization:
        # visualize the raw event images and the masks
        from PIL import Image
        import numpy as np
        event_pos_path = './event_pos.png'
        event_neg_path = './event_neg.png'
        mask_pos_path = './mask_pos.png'
        mask_neg_path = './mask_neg.png'

        event_pos_image = raw_event_pos[0].cpu().numpy()
        event_pos_image = Image.fromarray(event_pos_image, 'L')
        event_pos_image.save(event_pos_path)

        event_neg_image = raw_event_neg[0].cpu().numpy()
        event_neg_image = Image.fromarray(event_neg_image, 'L')
        event_neg_image.save(event_neg_path)

        mask_pos_image = event_pos_mask[0].cpu().numpy()
        mask_pos_image = 255 * (mask_pos_image - np.min(mask_pos_image)) / (np.max(mask_pos_image) - np.min(mask_pos_image))
        mask_pos_image = Image.fromarray(np.uint8(mask_pos_image), 'L')
        mask_pos_image.save(mask_pos_path)

        mask_neg_image = event_neg_mask[0].cpu().numpy()
        mask_neg_image = 255 * (mask_neg_image - np.min(mask_neg_image)) / (np.max(mask_neg_image) - np.min(mask_neg_image))
        mask_neg_image = Image.fromarray(np.uint8(mask_neg_image), 'L')
        mask_neg_image.save(mask_neg_path)

    return event_mask



def fine_grained_patch_distance(feat_1, feat_2, logit_scale=None):
    """
    feat_1, feat_2 of shape (N, seq_len, hid_dim, h, w)
    """
    b1, t1, hid_dim1, h1, w1 = feat_1.shape
    b2, t2, hid_dim2, h2, w2 = feat_2.shape
    assert t1 == t2
    assert hid_dim1 == hid_dim2
    assert h1 == h2
    assert w1 == w2
    feat_1 = feat_1 / feat_1.norm(dim=2, keepdim=True)
    feat_2 = feat_2 / feat_2.norm(dim=2, keepdim=True)
    if logit_scale is not None:
        scale = logit_scale.exp()
    else:
        scale = 1.0

    def get_logits(f1, f2):
        """
        Modified from https://github.com/Sense-GVT/DeCLIP/blob/main/prototype/model/filip.py
        """
        i, j, k, h1, w1 = f1.shape
        l, m, k, h2, w2 = f2.shape
        f1 = f1.permute(0, 1, 3, 4, 2)
        f2 = f2.permute(0, 1, 3, 4, 2)
        dense_feat_1 = f1.reshape(-1, k)
        dense_feat_2 = f2.reshape(-1, k)
        patch_distance_1 = get_patch_distance(dense_feat_1, dense_feat_2) # (ijh1w1, lmh2w2)
        patch_distance_2 = get_patch_distance(dense_feat_2, dense_feat_1) # (lmh2w2, ijh1w1)

        patch_distance_1 = patch_distance_1.reshape(i, j, h1*w1, l, m, h2*w2).max(dim=-1)[0].mean(dim=2) # (i,j,l,m)
        patch_distance_2 = patch_distance_2.reshape(l, m, h2*w2, i, j, h1*w1).max(dim=-1)[0].mean(dim=2).permute(2,3,0,1) # (l,m,i,j)
        logits_1 = 0.5 * scale * (patch_distance_1 + patch_distance_2)

        logits_1 = logits_1.reshape(i, j, l, m).permute(0, 2, 1, 3) #(i, l, j, m)
        return logits_1
    def get_patch_distance(f1, f2):
        i, k = f1.shape
        j, k = f2.shape
        patch_distance = f1 @ f2.t()
        return patch_distance


    final_logits_1 = get_logits(feat_1, feat_2).max(dim=-1)[0].mean(dim=-1) #(b1, b2)
    final_logits_2 = get_logits(feat_2, feat_1).max(dim=-1)[0].mean(dim=-1) #(b2, b1)

    final_logits = 0.5 * (final_logits_1 + final_logits_2.t()) # (b1, b2)

    return final_logits
