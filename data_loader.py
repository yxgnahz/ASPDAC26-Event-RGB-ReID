from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
from math import sqrt
import torch
from torch.utils.data import Dataset
import random
import math
import random
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage.transform import resize


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
        except EOFError:
            print("EOF incurred when reading '{}'. ".format(img_path))
            exit()
    return img


def read_image_edge(img_path, min=50, max=200):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    edges = cv2.Canny(image, min, max)
    return edges


class VideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (
                    S - num_ir)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])  # [selected_idx_for_interval_1, ...]
        number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

        imgs_ir = []
        for index in number_ir:
            index = int(index)
            img_path = img_ir_paths[index]

            img = read_image(img_path)
            # 添加
            img = np.array(img)

            if self.transform is not None:
                img = self.transform(img)

                # img = img.unsqueeze(0)
            imgs_ir.append(img)
        imgs_ir = torch.cat(imgs_ir, dim=0)  # [seq_len*c, h, w]

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            # 添加
            img = np.array(img)

            if self.transform is not None:
                img = self.transform(img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb


class VideoDataset_styletrain(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, transform_style=None,
                 index1=[], index2=[]):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.transform_style = transform_style
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]

        num_ir = len(img_ir_paths)

        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_ir.shape[1], sample_clip_ir.shape[0])
        number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

        imgs_ir = []
        imgs_ir_style = []
        for index in number_ir:
            index = int(index)
            img_path = img_ir_paths[index]

            img = read_image(img_path)

            img = np.array(img)

            if self.transform is not None:
                img = self.transform(img)
                img_style = self.transform_style(img)

            imgs_ir.append(img)
            imgs_ir_style.append(img_style)
        imgs_ir = torch.cat(imgs_ir, dim=0)
        imgs_ir_style = torch.cat(imgs_ir_style, dim=0)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]

        imgs_rgb = []
        imgs_rgb_style = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            img = np.array(img)

            if self.transform is not None:
                img = self.transform(img)
                imgs_style = self.transform_style(img)

            imgs_rgb.append(img)
            imgs_rgb_style.append(img)

        imgs_rgb = torch.cat(imgs_rgb, dim=0)
        imgs_rgb_style = torch.cat(imgs_rgb_style, dim=0)

        return imgs_ir, imgs_ir_style, pid_ir, camid_ir, imgs_rgb, imgs_rgb_style, pid_rgb, camid_rgb


class VideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    # 添加
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]

            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                mask_path = img_path[0:-4] + "_mask.npy"

                img = read_image(img_path)

                img = np.array(img)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class VideoDataset_train_evaluation(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, seq_len=12, sample='evenly', transform=None):
        self.dataset_ir = dataset_ir
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset_ir)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[index]

        num_ir = len(img_ir_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_ir_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            return imgs_ir, pid_ir, camid_ir
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class EventVideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_event_image(img_path)
                    # 添加
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_event_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]

            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                mask_path = img_path[0:-4] + "_mask.npy"

                img = read_event_image(img_path)

                img = np.array(img)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class EdgeEventVideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_event_image(img_path)
                    # 添加
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_event_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]

            imgs_ir = []
            imgs_edges = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                mask_path = img_path[0:-4] + "_mask.npy"

                img = read_event_image(img_path)
                if img_path[-4:] != '.npy':
                    edge_img = read_image_edge(img_path)
                    # edge img transform: duplicate to 3 channels and transform it 3-channel tensors
                    edge_img = np.repeat(edge_img[:, :, np.newaxis], 3, axis=2)
                    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB)
                    edge_img = self.transform(edge_img)
                    # edge_img = Image.fromarray(edge_img)
                    # edge_img = transforms.ToTensor()(edge_img)
                    imgs_edges.append(edge_img)

                img = np.array(img)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            if len(imgs_edges) > 0:
                imgs_edges = torch.cat(imgs_edges, dim=0)
            return imgs_ir, pid, camid, imgs_edges
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class EdgeFuseEventVideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_event_image(img_path)
                    # 添加
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_event_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]

            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                mask_path = img_path[0:-4] + "_mask.npy"
                # img = read_event_image(img_path)
                if img_path[-4:] != '.npy':
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    edge_img = read_image_edge(img_path)
                    # edge img transform: duplicate to 3 channels and transform it 3-channel tensors
                    edge_img = np.repeat(edge_img[:, :, np.newaxis], 3, axis=2)
                    img = cv2.add(img, edge_img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # edge_img = Image.fromarray(edge_img)
                    # edge_img = transforms.ToTensor()(edge_img)
                else:
                    img = read_event_image(img_path)
                img = np.array(img)

                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)

            return imgs_ir, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class MaskEventVideoDataset_test(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_event_image(img_path)
                    # 添加
                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            imgs_ir_raw = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_event_image(img_path)

                raw_img = np.array(img)
                imgs_ir_raw.append(raw_img)
                if self.transform is not None:
                    img = self.transform(raw_img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            imgs_ir_raw = np.stack(imgs_ir_raw)
            return imgs_ir, pid, camid, imgs_ir_raw

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]

            imgs_ir = []
            raw_imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                mask_path = img_path[0:-4] + "_mask.npy"

                img = read_event_image(img_path)

                raw_img = np.array(img)

                if self.transform is not None:
                    img = self.transform(raw_img)

                imgs_ir.append(img)
                raw_imgs_ir.append(raw_img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            raw_imgs_ir = torch.from_numpy(np.stack(raw_imgs_ir))
            return imgs_ir, pid, camid, raw_imgs_ir
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))


class FourierExchangeEventVideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_event, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[],
                 grayscale_trans=False, sep_trans=False, sep_trans_mixup=False, window_ratio=1.0, alpha=1.0,
                 sampling_mode='uniform'):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_event = dataset_event
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2
        self.grayscale_trans = grayscale_trans
        self.sep_trans = sep_trans
        self.sep_trans_mixup = sep_trans_mixup
        self.window_ratio = window_ratio
        self.alpha = alpha
        self.sample_mode = sampling_mode

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_event_paths, pid_event, camid_event = self.dataset_event[self.index2[index]]
        num_event = len(img_event_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_event = []
        frame_indices_event = list(range(num_event))
        if num_event < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (
                    S - num_event)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_event.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_event / S)
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (inter_val_ir * S - num_event)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_event.append(list(pool_ir))

        sample_clip_event = np.array(sample_clip_event)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_event.shape[1],
                                sample_clip_event.shape[0])  # [selected_idx_for_interval_1, ...]
        number_event = sample_clip_event[np.arange(len(sample_clip_event)), idx1]

        imgs_event = []
        raw_imgs_event = []
        for index in number_event:
            index = int(index)
            img_path = img_event_paths[index]
            img = read_event_image(img_path)
            raw_event_img = np.array(img)
            raw_imgs_event.append(raw_event_img)
            if self.transform is not None:
                img = self.transform(raw_event_img)

                # img = img.unsqueeze(0)
            imgs_event.append(img)
        imgs_event = torch.cat(imgs_event, dim=0)  # [seq_len*c, h, w]
        # raw_imgs_event = np.stack(raw_imgs_event, dim=0) # (seq_len, 3, h, w)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        raw_imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            raw_img = np.array(img)
            raw_imgs_rgb.append(raw_img)
            if self.transform is not None:
                img = self.transform(raw_img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
            # raw_imgs_rgb.append(raw_img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        # raw_imgs_rgb = np.stack(raw_imgs_rgb, dim=0)
        # fourier amplitude exchange
        exchanged_images_rgb, exchanged_images_event = self.fourier_amplitude_exchange(raw_imgs_rgb, raw_imgs_event)
        return imgs_event, pid_event, camid_event, imgs_rgb, pid_rgb, camid_rgb, exchanged_images_rgb, exchanged_images_event

    def fourier_amplitude_exchange(self, raw_imgs_rgb, raw_imgs_event):
        def to_grayscale(img):
            # Assuming img is a numpy array with shape (H, W, C)
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        assert len(raw_imgs_rgb) == len(raw_imgs_event)
        exchanged_images_rgb, exchanged_images_event = [], []

        for img_a, img_b in zip(raw_imgs_rgb, raw_imgs_event):

            if self.sep_trans_mixup:
                # from PIL import Image
                # print(f'event shape: {img_b.shape}')
                h_, w_, _ = img_b.shape
                # im_a = Image.fromarray(img_a)
                # im_a.save("img_a_raw.png")
                # print(f'rgb shape: {img_a.shape}')
                # print(f'max: {np.max(img_a)}')
                img_a = resize(img_a, (h_, w_, 3))
                img_a = (img_a * 255).astype(np.uint8)
                # print(f'rgb resized shape: {img_a.shape}')
                # print(f'max after resize: {np.max(img_a)}')
                # im_a = Image.fromarray(img_a)
                # im_a.save("img_a_resized.png")
                gray_raw = np.array(to_grayscale(img_a))
                pos_event, neg_event = np.array(img_b[:, :, 0]), np.array(img_b[:, :, 1])

                if self.sample_mode == 'uniform':
                    lam = np.random.uniform(0, self.alpha)
                elif self.sample_mode == 'fix':
                    lam = self.alpha
                else:
                    raise ValueError('sample mode unrecognized!')
                h, w = gray_raw.shape
                h_crop = int(h * sqrt(self.window_ratio))
                w_crop = int(w * sqrt(self.window_ratio))
                h_start = h // 2 - h_crop // 2
                w_start = w // 2 - w_crop // 2
                gray_raw_fft = np.fft.fft2(gray_raw, axes=(0, 1))
                pos_event_fft = np.fft.fft2(pos_event, axes=(0, 1))
                neg_event_fft = np.fft.fft2(neg_event, axes=(0, 1))
                gray_raw_abs, gray_raw_pha = np.abs(gray_raw_fft), np.angle(gray_raw_fft)
                pos_event_abs, pos_event_pha = np.abs(pos_event_fft), np.angle(pos_event_fft)
                neg_event_abs, neg_event_pha = np.abs(neg_event_fft), np.angle(neg_event_fft)

                gray_raw_abs = np.fft.fftshift(gray_raw_abs, axes=(0, 1))
                pos_event_abs = np.fft.fftshift(pos_event_abs, axes=(0, 1))
                neg_event_abs = np.fft.fftshift(neg_event_abs, axes=(0, 1))

                gray_raw_abs_ = np.copy(gray_raw_abs)
                pos_event_abs_ = np.copy(pos_event_abs)
                neg_event_abs_ = np.copy(neg_event_abs)

                gray_raw_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
                    lam * 0.5 * (pos_event_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + neg_event_abs_[
                                                                                                      h_start:h_start + h_crop,
                                                                                                      w_start:w_start + w_crop]) \
                    + (1 - lam) * gray_raw_abs_[h_start:h_start + h_crop,
                                  w_start:w_start + w_crop]

                pos_event_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
                    lam * gray_raw_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (
                                1 - lam) * pos_event_abs_[
                                           h_start:h_start + h_crop,
                                           w_start:w_start + w_crop]

                neg_event_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
                    lam * gray_raw_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (
                            1 - lam) * neg_event_abs_[
                                       h_start:h_start + h_crop,
                                       w_start:w_start + w_crop]

                gray_raw_abs = np.fft.ifftshift(gray_raw_abs, axes=(0, 1))
                pos_event_abs = np.fft.ifftshift(pos_event_abs, axes=(0, 1))
                neg_event_abs = np.fft.ifftshift(neg_event_abs, axes=(0, 1))

                exchanged_rgb = gray_raw_abs * (np.e ** (1j * gray_raw_pha))
                exchanged_event_pos = pos_event_abs * (np.e ** (1j * pos_event_pha))
                exchanged_event_neg = neg_event_abs * (np.e ** (1j * neg_event_pha))

                exchanged_rgb = np.real(np.fft.ifft2(exchanged_rgb, axes=(0, 1)))
                exchanged_event_pos = np.real(np.fft.ifft2(exchanged_event_pos, axes=(0, 1)))
                exchanged_event_neg = np.real(np.fft.ifft2(exchanged_event_neg, axes=(0, 1)))
                exchanged_rgb = np.repeat(exchanged_rgb[:, :, np.newaxis], 3, axis=2)
                exchanged_event = np.concatenate((exchanged_event_pos[:, :, np.newaxis], exchanged_event_neg[:, :, np.newaxis]), axis=2)
                exchanged_event = np.pad(exchanged_event, ((0, 0), (0, 0), (0, 1)), 'constant')
                exchanged_rgb = np.uint8(np.clip(exchanged_rgb, 0, 255))
                exchanged_event = np.uint8(np.clip(exchanged_event, 0, 255))
                # print(f'exchanged rgb shape: {exchanged_rgb.shape}')
                # print(f'exchanged event shape: {exchanged_event.shape}')
                # exit()
                # rgb_path_1 = 'MARS_rgb_sep_mixup_gray_raw.png'
                # event_pos_path_1 = 'MARS_event_pos_mixup_gray_raw.png'
                # event_neg_path_1 = 'MARS_event_neg_mixup_gray_raw.png'
                # rgb_path_2 = 'MARS_rgb_sep_mixup_gray_trans.png'
                # event_pos_path_2 = 'MARS_event_pos_mixup_gray_trans.png'
                # event_neg_path_2 = 'MARS_event_neg_mixup_gray_trans.png'
                # exchanged_rgb = Image.fromarray(exchanged_rgb)
                # exchanged_pos = Image.fromarray(np.uint8(np.clip(exchanged_event_pos, 0, 255)))
                # exchanged_neg = Image.fromarray(np.uint8(np.clip(exchanged_event_neg, 0, 255)))
                # Image.fromarray(gray_raw.astype('uint8')).save(rgb_path_1)
                # Image.fromarray(pos_event.astype('uint8')).save(event_pos_path_1)
                # Image.fromarray(neg_event.astype('uint8')).save(event_neg_path_1)
                # exchanged_rgb.save(rgb_path_2)
                # exchanged_pos.save(event_pos_path_2)
                # exchanged_neg.save(event_neg_path_2)
                # exit()
            elif self.sep_trans:
                gray_rgb = to_grayscale(img_a)
                pos_event, neg_event = img_b[:, :, 0], img_b[:, :, 1]
                gray_rgb_tensor = transforms.ToTensor()(gray_rgb.astype('uint8'))
                pos_event_tensor = transforms.ToTensor()(pos_event)
                neg_event_tensor = transforms.ToTensor()(neg_event)
                # print(f'gray rgb shape: {gray_rgb_tensor.shape} pos tensor shape: {pos_event_tensor.shape} {pos_event_tensor}')
                fft_rgb = torch.fft.fft2(gray_rgb_tensor)
                fft_pos = torch.fft.fft2(pos_event_tensor)
                fft_neg = torch.fft.fft2(neg_event_tensor)
                # print(f'fft rgb: {fft_rgb}')
                # print(f'fft neg: {fft_neg}')
                # Compute amplitude and phase
                amp_rgb, phase_rgb = torch.abs(fft_rgb), torch.angle(fft_rgb)
                amp_pos, phase_pos = torch.abs(fft_pos), torch.angle(fft_pos)
                amp_neg, phase_neg = torch.abs(fft_neg), torch.angle(fft_neg)
                # print(f'neg fft: {amp_neg} {phase_neg}')
                amp_event = 0.5 * (amp_neg + amp_pos)
                exchanged_rgb_fft = torch.polar(amp_event, phase_rgb)
                exchanged_pos_fft = torch.polar(amp_rgb, phase_pos)
                exchanged_neg_fft = torch.polar(amp_rgb, phase_neg)

                exchanged_rgb_ifft = torch.fft.ifft2(exchanged_rgb_fft).real.clamp(0, 1)
                exchanged_pos_ifft = torch.fft.ifft2(exchanged_pos_fft).real.clamp(0, 1)
                exchanged_neg_ifft = torch.fft.ifft2(exchanged_neg_fft).real.clamp(0, 1)
                exchanged_event = torch.cat([exchanged_pos_ifft, exchanged_neg_ifft], dim=0)
                exchanged_event = F.pad(exchanged_event, (0, 0, 0, 0, 0, 1), "constant", 0)
                exchanged_rgb = exchanged_rgb_ifft.repeat(3, 1, 1)

                # rgb_path_1 = 'rgb_sep_gray_raw.png'
                # event_pos_path_1 = 'event_pos_gray_raw.png'
                # event_neg_path_1 = 'event_neg_gray_raw.png'
                # rgb_path_2 = 'rgb_sep_gray_trans.png'
                # event_pos_path_2 = 'event_pos_gray_trans.png'
                # event_neg_path_2 = 'event_neg_gray_trans.png'
                # exchanged_rgb = transforms.ToPILImage()(exchanged_rgb_ifft)
                # exchanged_pos = transforms.ToPILImage()(exchanged_pos_ifft)
                # exchanged_neg = transforms.ToPILImage()(exchanged_neg_ifft)
                # Image.fromarray(gray_rgb.astype('uint8')).save(rgb_path_1)
                # Image.fromarray(pos_event.astype('uint8')).save(event_pos_path_1)
                # Image.fromarray(neg_event.astype('uint8')).save(event_neg_path_1)
                # exchanged_rgb.save(rgb_path_2)
                # exchanged_pos.save(event_pos_path_2)
                # exchanged_neg.save(event_neg_path_2)

            elif self.grayscale_trans:
                # transform the images to grayscale first then conduct FFT

                # Convert to grayscale
                gray_a = to_grayscale(img_a).astype('uint8')
                gray_b = to_grayscale(img_b).astype('uint8')

                # Convert numpy arrays to PyTorch tensors and add a batch dimension
                tensor_a = transforms.ToTensor()(gray_a)  # Add channel and batch dimension
                tensor_b = transforms.ToTensor()(gray_b)

                # Apply 2D FFT to the grayscale images
                fft_a = torch.fft.fft2(tensor_a)
                fft_b = torch.fft.fft2(tensor_b)

                # Compute amplitude and phase
                amp_a, phase_a = torch.abs(fft_a), torch.angle(fft_a)
                amp_b, phase_b = torch.abs(fft_b), torch.angle(fft_b)

                # Exchange amplitudes
                exchanged_fft_a = torch.polar(amp_b, phase_a)
                exchanged_fft_b = torch.polar(amp_a, phase_b)

                # Apply inverse 2D FFT
                exchanged_rgb_ifft = torch.fft.ifft2(exchanged_fft_a).real.clamp(0, 1)
                exchanged_event_ifft = torch.fft.ifft2(exchanged_fft_b).real.clamp(0, 1)

                # Expand the grayscale image to RGB
                exchanged_rgb = exchanged_rgb_ifft.repeat(3, 1, 1)
                exchanged_event = exchanged_event_ifft.repeat(3, 1, 1)

                # rgb_path_1 = 'rgb_gray_raw.png'
                # event_path_1 = 'event_gray_raw.png'
                # rgb_path_2 = 'rgb_gray_trans.png'
                # event_path_2 = 'event_gray_trans.png'
                # Image.fromarray(gray_a).save(rgb_path_1)
                # Image.fromarray(gray_b).save(event_path_1)
                # Image.fromarray(exchanged_img_a.astype('uint8')).save(rgb_path_2)
                # Image.fromarray(exchanged_img_b.astype('uint8')).save(event_path_2)
            else:
                # Convert numpy arrays to PyTorch tensors and add a batch dimension
                tensor_a = transforms.ToTensor()(img_a.astype('uint8')).float()
                tensor_b = transforms.ToTensor()(img_b.astype('uint8')).float()
                # Apply 2D FFT to each channel
                fft_a = torch.fft.fft2(tensor_a, dim=(1, 2))
                fft_b = torch.fft.fft2(tensor_b, dim=(1, 2))
                # Compute amplitude and phase
                amp_a, phase_a = torch.abs(fft_a), torch.angle(fft_a)
                amp_b, phase_b = torch.abs(fft_b), torch.angle(fft_b)
                # Exchange amplitudes
                exchanged_fft_a = torch.polar(amp_b, phase_a)
                exchanged_fft_b = torch.polar(amp_a, phase_b)

                # Apply inverse 2D FFT
                exchanged_rgb = torch.fft.ifft2(exchanged_fft_a, dim=(1, 2)).real.clamp(0, 1)
                exchanged_event = torch.fft.ifft2(exchanged_fft_b, dim=(1, 2)).real.clamp(0, 1)
                print(f'exchanged rgb event shape: {exchanged_rgb.shape} {exchanged_event.shape}')

            # Append to the list of exchanged images
            exchanged_rgb = self.transform(exchanged_rgb)
            exchanged_event = self.transform(exchanged_event)
            exchanged_images_rgb.append(exchanged_rgb)
            exchanged_images_event.append(exchanged_event)
        exchanged_images_rgb = torch.cat(exchanged_images_rgb, dim=0)
        exchanged_images_event = torch.cat(exchanged_images_event, dim=0)

        return exchanged_images_rgb, exchanged_images_event


class FourierExchangThermalVideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_ir, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[],
                 grayscale_trans=False, sep_trans=False, sep_trans_mixup=False, window_ratio=1.0, alpha=1.0,
                 sampling_mode='uniform'):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2
        self.grayscale_trans = grayscale_trans
        self.sep_trans = sep_trans
        self.sep_trans_mixup = sep_trans_mixup
        self.window_ratio = window_ratio
        self.alpha = alpha
        self.sample_mode = sampling_mode

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        num_ir = len(img_ir_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num_ir))
        if num_ir < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (
                    S - num_ir)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_ir / S)
            strip_ir = list(range(num_ir)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num_ir)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_ir.shape[1],
                                sample_clip_ir.shape[0])  # [selected_idx_for_interval_1, ...]
        number_ir = sample_clip_ir[np.arange(len(sample_clip_ir)), idx1]

        imgs_ir = []
        raw_imgs_ir = []
        for index in number_ir:
            index = int(index)
            img_path = img_ir_paths[index]
            img = read_image(img_path)
            raw_ir_img = np.array(img)
            raw_imgs_ir.append(raw_ir_img)
            if self.transform is not None:
                img = self.transform(raw_ir_img)

                # img = img.unsqueeze(0)
            imgs_ir.append(img)
        imgs_ir = torch.cat(imgs_ir, dim=0)  # [seq_len*c, h, w]
        # raw_imgs_event = np.stack(raw_imgs_event, dim=0) # (seq_len, 3, h, w)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        raw_imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            raw_img = np.array(img)
            raw_imgs_rgb.append(raw_img)
            if self.transform is not None:
                img = self.transform(raw_img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
            # raw_imgs_rgb.append(raw_img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        # raw_imgs_rgb = np.stack(raw_imgs_rgb, dim=0)
        # fourier amplitude exchange
        exchanged_images_rgb, exchanged_images_ir = self.fourier_amplitude_exchange(raw_imgs_rgb, raw_imgs_ir)
        return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb, exchanged_images_rgb, exchanged_images_ir

    def fourier_amplitude_exchange(self, raw_imgs_rgb, raw_imgs_ir):
        def to_grayscale(img):
            # Assuming img is a numpy array with shape (H, W, C)
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        assert len(raw_imgs_rgb) == len(raw_imgs_ir)
        exchanged_images_rgb, exchanged_images_ir = [], []

        for img_a, img_b in zip(raw_imgs_rgb, raw_imgs_ir):
            # print(f'img a b data type: {img_a.dtype} {img_b.dtype}')
            if self.sep_trans_mixup:
                # from PIL import Image
                # print(f'event shape: {img_b.shape}')
                h_, w_, _ = img_b.shape
                # im_a = Image.fromarray(img_a)
                # im_a.save("img_a_raw.png")
                # print(f'rgb shape: {img_a.shape}')
                # print(f'max: {np.max(img_a)}')
                img_a = resize(img_a, (h_, w_, 3))
                img_a = (img_a * 255).astype(np.uint8)
                # print(f'rgb resized shape: {img_a.shape}')
                # print(f'max after resize: {np.max(img_a)}')
                # im_a = Image.fromarray(img_a)
                # im_a.save("img_a_resized.png")
                gray_raw = np.array(to_grayscale(img_a))
                ir_raw = np.array(img_b[:, :, 0].astype(np.float32))
                # print(f'max after resize: {np.max(gray_raw)} {np.max(ir_raw)}')

                if self.sample_mode == 'uniform':
                    lam = np.random.uniform(0, self.alpha)
                elif self.sample_mode == 'fix':
                    lam = self.alpha
                else:
                    raise ValueError('sample mode unrecognized!')
                h, w = gray_raw.shape
                h_crop = int(h * sqrt(self.window_ratio))
                w_crop = int(w * sqrt(self.window_ratio))
                h_start = h // 2 - h_crop // 2
                w_start = w // 2 - w_crop // 2
                gray_raw_fft = np.fft.fft2(gray_raw, axes=(0, 1))
                ir_raw_fft = np.fft.fft2(ir_raw, axes=(0, 1))
                gray_raw_abs, gray_raw_pha = np.abs(gray_raw_fft), np.angle(gray_raw_fft)
                ir_raw_abs, ir_raw_pha = np.abs(ir_raw_fft), np.angle(ir_raw_fft)
                gray_raw_abs = np.fft.fftshift(gray_raw_abs, axes=(0, 1))
                ir_raw_abs = np.fft.fftshift(ir_raw_abs, axes=(0, 1))

                gray_raw_abs_ = np.copy(gray_raw_abs)
                ir_raw_abs_ = np.copy(ir_raw_abs)

                gray_raw_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
                    lam * (ir_raw_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]) \
                    + (1 - lam) * gray_raw_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]

                ir_raw_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
                    lam * gray_raw_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (
                            1 - lam) * ir_raw_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]

                gray_raw_abs = np.fft.ifftshift(gray_raw_abs, axes=(0, 1))
                ir_raw_abs = np.fft.ifftshift(ir_raw_abs, axes=(0, 1))

                exchanged_rgb = gray_raw_abs * (np.e ** (1j * gray_raw_pha))
                exchanged_ir = ir_raw_abs * (np.e ** (1j * ir_raw_pha))

                exchanged_rgb = np.real(np.fft.ifft2(exchanged_rgb, axes=(0, 1)))
                exchanged_ir = np.real(np.fft.ifft2(exchanged_ir, axes=(0, 1)))

                exchanged_rgb = np.repeat(exchanged_rgb[:, :, np.newaxis], 3, axis=2)
                exchanged_ir = np.repeat(exchanged_ir[:, :, np.newaxis], 3, axis=2)

                exchanged_rgb = np.uint8(np.clip(exchanged_rgb, 0, 255))
                exchanged_ir = np.uint8(np.clip(exchanged_ir, 0, 255))
                # print(f'exchanged rgb shape: {exchanged_rgb.shape}')
                # print(f'exchanged event shape: {exchanged_event.shape}')
                # exit()
                # rgb_path_1 = 'MARS_rgb_sep_mixup_gray_raw.png'
                # event_pos_path_1 = 'MARS_event_pos_mixup_gray_raw.png'
                # event_neg_path_1 = 'MARS_event_neg_mixup_gray_raw.png'
                # rgb_path_2 = 'MARS_rgb_sep_mixup_gray_trans.png'
                # event_pos_path_2 = 'MARS_event_pos_mixup_gray_trans.png'
                # event_neg_path_2 = 'MARS_event_neg_mixup_gray_trans.png'
                # exchanged_rgb = Image.fromarray(exchanged_rgb)
                # exchanged_pos = Image.fromarray(np.uint8(np.clip(exchanged_event_pos, 0, 255)))
                # exchanged_neg = Image.fromarray(np.uint8(np.clip(exchanged_event_neg, 0, 255)))
                # Image.fromarray(gray_raw.astype('uint8')).save(rgb_path_1)
                # Image.fromarray(pos_event.astype('uint8')).save(event_pos_path_1)
                # Image.fromarray(neg_event.astype('uint8')).save(event_neg_path_1)
                # exchanged_rgb.save(rgb_path_2)
                # exchanged_pos.save(event_pos_path_2)
                # exchanged_neg.save(event_neg_path_2)
                # exit()
            elif self.sep_trans:
                gray_rgb = to_grayscale(img_a)
                pos_event, neg_event = img_b[:, :, 0], img_b[:, :, 1]
                gray_rgb_tensor = transforms.ToTensor()(gray_rgb.astype('uint8'))
                pos_event_tensor = transforms.ToTensor()(pos_event)
                neg_event_tensor = transforms.ToTensor()(neg_event)
                # print(f'gray rgb shape: {gray_rgb_tensor.shape} pos tensor shape: {pos_event_tensor.shape} {pos_event_tensor}')
                fft_rgb = torch.fft.fft2(gray_rgb_tensor)
                fft_pos = torch.fft.fft2(pos_event_tensor)
                fft_neg = torch.fft.fft2(neg_event_tensor)
                # print(f'fft rgb: {fft_rgb}')
                # print(f'fft neg: {fft_neg}')
                # Compute amplitude and phase
                amp_rgb, phase_rgb = torch.abs(fft_rgb), torch.angle(fft_rgb)
                amp_pos, phase_pos = torch.abs(fft_pos), torch.angle(fft_pos)
                amp_neg, phase_neg = torch.abs(fft_neg), torch.angle(fft_neg)
                # print(f'neg fft: {amp_neg} {phase_neg}')
                amp_event = 0.5 * (amp_neg + amp_pos)
                exchanged_rgb_fft = torch.polar(amp_event, phase_rgb)
                exchanged_pos_fft = torch.polar(amp_rgb, phase_pos)
                exchanged_neg_fft = torch.polar(amp_rgb, phase_neg)

                exchanged_rgb_ifft = torch.fft.ifft2(exchanged_rgb_fft).real.clamp(0, 1)
                exchanged_pos_ifft = torch.fft.ifft2(exchanged_pos_fft).real.clamp(0, 1)
                exchanged_neg_ifft = torch.fft.ifft2(exchanged_neg_fft).real.clamp(0, 1)
                exchanged_event = torch.cat([exchanged_pos_ifft, exchanged_neg_ifft], dim=0)
                exchanged_event = F.pad(exchanged_event, (0, 0, 0, 0, 0, 1), "constant", 0)
                exchanged_rgb = exchanged_rgb_ifft.repeat(3, 1, 1)

                # rgb_path_1 = 'rgb_sep_gray_raw.png'
                # event_pos_path_1 = 'event_pos_gray_raw.png'
                # event_neg_path_1 = 'event_neg_gray_raw.png'
                # rgb_path_2 = 'rgb_sep_gray_trans.png'
                # event_pos_path_2 = 'event_pos_gray_trans.png'
                # event_neg_path_2 = 'event_neg_gray_trans.png'
                # exchanged_rgb = transforms.ToPILImage()(exchanged_rgb_ifft)
                # exchanged_pos = transforms.ToPILImage()(exchanged_pos_ifft)
                # exchanged_neg = transforms.ToPILImage()(exchanged_neg_ifft)
                # Image.fromarray(gray_rgb.astype('uint8')).save(rgb_path_1)
                # Image.fromarray(pos_event.astype('uint8')).save(event_pos_path_1)
                # Image.fromarray(neg_event.astype('uint8')).save(event_neg_path_1)
                # exchanged_rgb.save(rgb_path_2)
                # exchanged_pos.save(event_pos_path_2)
                # exchanged_neg.save(event_neg_path_2)

            elif self.grayscale_trans:
                # transform the images to grayscale first then conduct FFT

                # Convert to grayscale
                gray_a = to_grayscale(img_a).astype('uint8')
                gray_b = to_grayscale(img_b).astype('uint8')

                # Convert numpy arrays to PyTorch tensors and add a batch dimension
                tensor_a = transforms.ToTensor()(gray_a)  # Add channel and batch dimension
                tensor_b = transforms.ToTensor()(gray_b)

                # Apply 2D FFT to the grayscale images
                fft_a = torch.fft.fft2(tensor_a)
                fft_b = torch.fft.fft2(tensor_b)

                # Compute amplitude and phase
                amp_a, phase_a = torch.abs(fft_a), torch.angle(fft_a)
                amp_b, phase_b = torch.abs(fft_b), torch.angle(fft_b)

                # Exchange amplitudes
                exchanged_fft_a = torch.polar(amp_b, phase_a)
                exchanged_fft_b = torch.polar(amp_a, phase_b)

                # Apply inverse 2D FFT
                exchanged_rgb_ifft = torch.fft.ifft2(exchanged_fft_a).real.clamp(0, 1)
                exchanged_event_ifft = torch.fft.ifft2(exchanged_fft_b).real.clamp(0, 1)

                # Expand the grayscale image to RGB
                exchanged_rgb = exchanged_rgb_ifft.repeat(3, 1, 1)
                exchanged_event = exchanged_event_ifft.repeat(3, 1, 1)

                # rgb_path_1 = 'rgb_gray_raw.png'
                # event_path_1 = 'event_gray_raw.png'
                # rgb_path_2 = 'rgb_gray_trans.png'
                # event_path_2 = 'event_gray_trans.png'
                # Image.fromarray(gray_a).save(rgb_path_1)
                # Image.fromarray(gray_b).save(event_path_1)
                # Image.fromarray(exchanged_img_a.astype('uint8')).save(rgb_path_2)
                # Image.fromarray(exchanged_img_b.astype('uint8')).save(event_path_2)
            else:
                # Convert numpy arrays to PyTorch tensors and add a batch dimension
                tensor_a = transforms.ToTensor()(img_a.astype('uint8')).float()
                tensor_b = transforms.ToTensor()(img_b.astype('uint8')).float()
                # Apply 2D FFT to each channel
                fft_a = torch.fft.fft2(tensor_a, dim=(1, 2))
                fft_b = torch.fft.fft2(tensor_b, dim=(1, 2))
                # Compute amplitude and phase
                amp_a, phase_a = torch.abs(fft_a), torch.angle(fft_a)
                amp_b, phase_b = torch.abs(fft_b), torch.angle(fft_b)
                # Exchange amplitudes
                exchanged_fft_a = torch.polar(amp_b, phase_a)
                exchanged_fft_b = torch.polar(amp_a, phase_b)

                # Apply inverse 2D FFT
                exchanged_rgb = torch.fft.ifft2(exchanged_fft_a, dim=(1, 2)).real.clamp(0, 1)
                exchanged_event = torch.fft.ifft2(exchanged_fft_b, dim=(1, 2)).real.clamp(0, 1)
                print(f'exchanged rgb event shape: {exchanged_rgb.shape} {exchanged_event.shape}')

            # Append to the list of exchanged images
            exchanged_rgb = self.transform(exchanged_rgb)
            exchanged_ir = self.transform(exchanged_ir)
            exchanged_images_rgb.append(exchanged_rgb)
            exchanged_images_ir.append(exchanged_ir)
        exchanged_images_rgb = torch.cat(exchanged_images_rgb, dim=0)
        exchanged_images_ir = torch.cat(exchanged_images_ir, dim=0)

        return exchanged_images_rgb, exchanged_images_ir

class EventVideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_event, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_event = dataset_event
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_event_paths, pid_event, camid_event = self.dataset_event[self.index2[index]]
        num_event = len(img_event_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_event = []
        frame_indices_event = list(range(num_event))
        if num_event < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (
                    S - num_event)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_event.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_event / S)
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (inter_val_ir * S - num_event)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_event.append(list(pool_ir))

        sample_clip_event = np.array(sample_clip_event)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_event.shape[1],
                                sample_clip_event.shape[0])  # [selected_idx_for_interval_1, ...]
        number_event = sample_clip_event[np.arange(len(sample_clip_event)), idx1]

        imgs_event = []
        # raw_imgs_event = []
        for index in number_event:
            index = int(index)
            img_path = img_event_paths[index]

            img = read_event_image(img_path)
            # 添加
            raw_event_img = np.array(img)
            # raw_imgs_event.append(raw_event_img)
            if self.transform is not None:
                img = self.transform(raw_event_img)

                # img = img.unsqueeze(0)
            imgs_event.append(img)
        imgs_event = torch.cat(imgs_event, dim=0)  # [seq_len*c, h, w]
        # raw_imgs_event = np.stack(raw_imgs_event, dim=0) # (seq_len, 3, h, w)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        # raw_imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            # 添加
            raw_img = np.array(img)

            if self.transform is not None:
                img = self.transform(raw_img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
            # raw_imgs_rgb.append(raw_img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        # raw_imgs_rgb = np.stack(raw_imgs_rgb, dim=0)
        return imgs_event, pid_event, camid_event, imgs_rgb, pid_rgb, camid_rgb


class EdgeEventVideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_event, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_event = dataset_event
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_event_paths, pid_event, camid_event = self.dataset_event[self.index2[index]]
        num_event = len(img_event_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_event = []
        frame_indices_event = list(range(num_event))
        if num_event < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (
                    S - num_event)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_event.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_event / S)
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (inter_val_ir * S - num_event)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_event.append(list(pool_ir))

        sample_clip_event = np.array(sample_clip_event)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_event.shape[1],
                                sample_clip_event.shape[0])  # [selected_idx_for_interval_1, ...]
        number_event = sample_clip_event[np.arange(len(sample_clip_event)), idx1]

        imgs_event = []
        # raw_imgs_event = []
        for index in number_event:
            index = int(index)
            img_path = img_event_paths[index]

            img = read_event_image(img_path)
            # 添加
            raw_event_img = np.array(img)
            # raw_imgs_event.append(raw_event_img)
            if self.transform is not None:
                img = self.transform(raw_event_img)

                # img = img.unsqueeze(0)
            imgs_event.append(img)
        imgs_event = torch.cat(imgs_event, dim=0)  # [seq_len*c, h, w]
        # raw_imgs_event = np.stack(raw_imgs_event, dim=0) # (seq_len, 3, h, w)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        imgs_edges = []
        # raw_imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            edge_img = read_image_edge(img_path)

            # edge img transform: duplicate to 3 channels and transform it 3-channel tensors
            edge_img = np.repeat(edge_img[:, :, np.newaxis], 3, axis=2)
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB)
            edge_img = self.transform(edge_img)
            # edge_img = Image.fromarray(edge_img)
            # edge_img = transforms.ToTensor()(edge_img)
            imgs_edges.append(edge_img)
            # 添加
            raw_img = np.array(img)
            if self.transform is not None:
                img = self.transform(raw_img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
            # raw_imgs_rgb.append(raw_img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        imgs_edge = torch.cat(imgs_edges, dim=0)
        # raw_imgs_rgb = np.stack(raw_imgs_rgb, dim=0)
        return imgs_event, pid_event, camid_event, imgs_rgb, pid_rgb, camid_rgb, imgs_edge


class EdgeFuseEventVideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_event, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_event = dataset_event
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_event_paths, pid_event, camid_event = self.dataset_event[self.index2[index]]
        num_event = len(img_event_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_event = []
        frame_indices_event = list(range(num_event))
        if num_event < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (
                    S - num_event)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_event.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_event / S)
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (inter_val_ir * S - num_event)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_event.append(list(pool_ir))

        sample_clip_event = np.array(sample_clip_event)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_event.shape[1],
                                sample_clip_event.shape[0])  # [selected_idx_for_interval_1, ...]
        number_event = sample_clip_event[np.arange(len(sample_clip_event)), idx1]

        imgs_event = []
        # raw_imgs_event = []
        for index in number_event:
            index = int(index)
            img_path = img_event_paths[index]

            img = read_event_image(img_path)
            # 添加
            raw_event_img = np.array(img)
            # raw_imgs_event.append(raw_event_img)
            if self.transform is not None:
                img = self.transform(raw_event_img)

                # img = img.unsqueeze(0)
            imgs_event.append(img)
        imgs_event = torch.cat(imgs_event, dim=0)  # [seq_len*c, h, w]
        # raw_imgs_event = np.stack(raw_imgs_event, dim=0) # (seq_len, 3, h, w)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            edge_img = read_image_edge(img_path)
            # edge img transform: duplicate to 3 channels and transform it 3-channel tensors
            edge_img = np.repeat(edge_img[:, :, np.newaxis], 3, axis=2)
            img = cv2.add(img, edge_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                img = self.transform(img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
            # raw_imgs_rgb.append(raw_img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        # raw_imgs_rgb = np.stack(raw_imgs_rgb, dim=0)
        return imgs_event, pid_event, camid_event, imgs_rgb, pid_rgb, camid_rgb


class MaskEventVideoDataset_train(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset_event, dataset_rgb, seq_len=12, sample='evenly', transform=None, index1=[], index2=[]):
        """
        seq_len denotes the number of frames for each video tracklet
        """
        self.dataset_event = dataset_event
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = index1
        self.index2 = index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):

        img_event_paths, pid_event, camid_event = self.dataset_event[self.index2[index]]
        num_event = len(img_event_paths)
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]
        num_rgb = len(img_rgb_paths)

        S = self.seq_len
        sample_clip_event = []
        frame_indices_event = list(range(num_event))
        if num_event < S:  # every tracklet is divided into 8 parts, randomly select one frame from each part.
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (
                    S - num_event)  # padding to seq len with the last frame
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_event.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num_event / S)
            strip_ir = list(range(num_event)) + [frame_indices_event[-1]] * (inter_val_ir * S - num_event)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_event.append(list(pool_ir))

        sample_clip_event = np.array(sample_clip_event)  # S * n_interval

        sample_clip_rgb = []
        frame_indices_rgb = list(range(num_rgb))
        if num_rgb < S:
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[s * 1:(s + 1) * 1]
                sample_clip_rgb.append(list(pool_rgb))
        else:
            inter_val_rgb = math.ceil(num_rgb / S)
            strip_rgb = list(range(num_rgb)) + [frame_indices_rgb[-1]] * (inter_val_rgb * S - num_rgb)
            for s in range(S):
                pool_rgb = strip_rgb[inter_val_rgb * s:inter_val_rgb * (s + 1)]
                sample_clip_rgb.append(list(pool_rgb))

        sample_clip_rgb = np.array(sample_clip_rgb)

        idx1 = np.random.choice(sample_clip_event.shape[1],
                                sample_clip_event.shape[0])  # [selected_idx_for_interval_1, ...]
        number_event = sample_clip_event[np.arange(len(sample_clip_event)), idx1]

        imgs_event = []
        raw_imgs_event = []
        for index in number_event:
            index = int(index)
            img_path = img_event_paths[index]

            img = read_event_image(img_path)
            # 添加
            raw_event_img = np.array(img)

            if self.transform is not None:
                img = self.transform(raw_event_img)
                # img = img.unsqueeze(0)
            imgs_event.append(img)
            raw_event_img = np.transpose(raw_event_img, (2, 0, 1))
            raw_imgs_event.append(raw_event_img)
        imgs_event = torch.cat(imgs_event, dim=0)  # [seq_len*c, h, w]
        raw_imgs_event = np.stack(raw_imgs_event)  # (seq_len, 3, h, w)

        idx2 = np.random.choice(sample_clip_rgb.shape[1], sample_clip_rgb.shape[0])
        number_rgb = sample_clip_rgb[np.arange(len(sample_clip_rgb)), idx2]
        imgs_rgb = []
        raw_imgs_rgb = []
        for index in number_rgb:
            index = int(index)
            img_path = img_rgb_paths[index]

            img = read_image(img_path)
            # 添加
            raw_img = np.array(img)
            if self.transform is not None:
                img = self.transform(raw_img)
            # img = img.unsqueeze(0)
            imgs_rgb.append(img)
            raw_img = np.transpose(raw_img, (2, 0, 1))
            raw_imgs_rgb.append(raw_img)
        imgs_rgb = torch.cat(imgs_rgb, dim=0)  # (seq_len*num_channel, )
        raw_imgs_rgb = torch.from_numpy(np.stack(raw_imgs_rgb))
        # print(f'raw rgb shape: {raw_imgs_rgb.shape}')
        # print(f'raw event shape: {raw_imgs_event.shape}')
        # exit()
        return imgs_event, pid_event, camid_event, imgs_rgb, pid_rgb, camid_rgb, raw_imgs_event, raw_imgs_rgb


def read_event_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            if img_path[-4:] == '.npy':
                img = np.load(img_path)
                # print(f'event sample: {img}')
                """
                By default we padding zeros to the third channel of the event img, to make it compatible to the images
                """
                padding = ((0, 0), (0, 0), (0, 1))
                # Apply padding
                img = np.pad(img, pad_width=padding, mode='constant', constant_values=0)
                # Binarize the image
                img = np.where(img >= 1, 1, 0)
                img *= 255
                img = img.astype(np.uint8)
                # print(f'event img sample: {img}')
                # print(f'event img shape: {img.shape}')
                # exit()
            else:
                img = Image.open(img_path).convert('RGB')
                # print(f'img sample: {img}')
                # print(f'img shape: {img.shape}')
                # exit()
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
        except EOFError:
            print("EOF incurred when reading '{}'. ".format(img_path))
            exit()
    return img
