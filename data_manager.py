from __future__ import print_function, absolute_import
import os
import os.path as osp
import numpy as np
import json


'''
def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    camera = fname[6:8]
    F_pos = fname.find('F')
    picture = fname[F_pos+1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path
'''

def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    
    # D/T/F identify a certain frame
    # D=camera id
    # F=frame id
    # T=tracklet id

    T_pos = fname.find('T')
    D_pos = fname.find('D')
    F_pos = fname.find('F')
    camera = fname[D_pos:T_pos]
    picture = fname[F_pos+1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path


class VCM(object):
    root = '/research/d4/gds/xyzhang21/event-camera/SAADG_VVIReID/data/VCM-HITSZ'

    # training data
    train_name_path = osp.join(root, 'info/train_name.txt')
    track_train_info_path = osp.join(root, 'info/track_train_info.txt')

    # testing data
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_test_info_path = osp.join(root, 'info/track_test_info.txt')

    query_IDX_path = osp.join(root, 'info/query_IDX.txt')


    def __init__(self, min_seq_len=12):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)  # image lists in training set
        track_train = self._get_tracks(self.track_train_info_path)  # tracklet basic information

        # for test
        test_names = self._get_names(self.test_name_path)
        track_test = self._get_tracks(self.track_test_info_path) # np.array

        query_IDX = self._get_query_idx(self.query_IDX_path) # np.array
        query_IDX -= 1

        track_query = track_test[query_IDX, :]

        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        #---------visible to infrared-----------
        gallery_IDX_1 = self._get_query_idx(self.query_IDX_path)
        gallery_IDX_1 -= 1
        track_gallery_1 = track_test[gallery_IDX_1, :]

        query_IDX_1 = [j for j in range(track_test.shape[0]) if j not in gallery_IDX_1]
        track_query_1 = track_test[query_IDX_1, :]
        #-----------------------------------------

        train_ir, num_train_tracklets_ir,num_train_imgs_ir,train_rgb, num_train_tracklets_rgb,num_train_imgs_rgb,num_train_pids,ir_label,rgb_label = \
          self._process_data_train(train_names,track_train,relabel=True,min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data_test(test_names, track_query, relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data_test(test_names, track_gallery, relabel=False, min_seq_len=min_seq_len)

        #--------visible to infrared-----------
        query_1, num_query_tracklets_1, num_query_pids_1, num_query_imgs_1 = \
          self._process_data_test(test_names, track_query_1, relabel=False, min_seq_len=min_seq_len)

        gallery_1, num_gallery_tracklets_1, num_gallery_pids_1, num_gallery_imgs_1 = \
          self._process_data_test(test_names, track_gallery_1, relabel=False, min_seq_len=min_seq_len)
        #---------------------------------------

        print("=> VCM loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_ir))
        print("train_rgb   | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_rgb))
        print("query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("---------------------------------")

        self.train_ir = train_ir
        self.train_rgb = train_rgb
        self.ir_label = ir_label
        self.rgb_label = rgb_label

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets
        self.num_gallery_tracklets = num_gallery_tracklets

        #------- visible to infrared------------
        self.query_1 = query_1
        self.gallery_1 = gallery_1
        self.num_train_tracklets_ir = num_train_tracklets_ir
        self.num_train_tracklets_rgb = num_train_tracklets_rgb

        self.num_query_pids_1 = num_query_pids_1
        self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_1
        self.num_gallery_tracklets_1 = num_gallery_tracklets_1
        #---------------------------------------


    def _check_before_run(self):
        """check before run """
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))

    def _get_names(self, fpath):
        """get image name, retuen name list"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _get_tracks(self, fpath):
        """get tracks file"""
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                names.append(tmp)
        names = np.array(names)
        return names

    '''
    def _get_query_idx(self,fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')
                # print(new_line.split(' ')[0:-1])
                tmp = new_line.split(' ')[0:-1]
                if new_line.split(' ')[-1][-1] == '1':
                    tmp.append(new_line.split(' ')[-1][0] + '1')
                else:
                    tmp.append(new_line.split(' ')[-1][0] + '2')
                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        print(idxs)
        return idxs
    '''

    def _get_query_idx(self, fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')
                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        # print(idxs)
        return idxs

    def _process_data_train(self,names,meta_data,relabel=False,min_seq_len=0):
        # meta data: tracklets list
        # every tracklet = [modality label, start frame id, end frame id, pid, camid]
        # "start frame id" starts from 1 and increases by 24 each time until 232458.

        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 3].tolist()))
        num_pids = len(pid_list)

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets_ir = []
        num_imgs_per_tracklet_ir = []
        ir_label = []

        tracklets_rgb = []
        num_imgs_per_tracklet_rgb = []
        rgb_label = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            m, start_index, end_index, pid, camid = data
            if relabel:
                pid = pid2label[pid]

            if m == 1:  # IR modality 
                # The "names" stores information for 230405 images in the format of [pid,modality label,cam id, tracklet id, image id]
  
                img_names = names[start_index-1:end_index]
                img_ir_paths = [osp.join(self.root, "Train", decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_ir_paths) >= min_seq_len:  # Filter out samples with low frame rates
                    img_ir_paths = tuple(img_ir_paths)  # The paths of all frames in a tracklet
                    ir_label.append(pid)
                    tracklets_ir.append((img_ir_paths, pid, camid))
                    # same id
                    num_imgs_per_tracklet_ir.append(len(img_ir_paths)) 
            else:
                img_names = names[start_index-1:end_index]
                img_rgb_paths = [osp.join(self.root, "Train", decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_rgb_paths) >= min_seq_len:
                    img_rgb_paths = tuple(img_rgb_paths)
                    rgb_label.append(pid)
                    tracklets_rgb.append((img_rgb_paths,pid,camid))
                    # same id
                    num_imgs_per_tracklet_rgb.append(len(img_rgb_paths))

        num_tracklets_ir = len(tracklets_ir)  # 4291
        num_tracklets_rgb = len(tracklets_rgb)  # 5460

        num_tracklets = num_tracklets_rgb  + num_tracklets_ir  #9751

        return tracklets_ir, num_tracklets_ir,num_imgs_per_tracklet_ir,tracklets_rgb,num_tracklets_rgb,num_imgs_per_tracklet_rgb,num_pids,ir_label,rgb_label


    def _process_data_test(self,names,meta_data,relabel=False,min_seq_len=0):
        # meta_data format
        # [1 284 307 503 4]
        # 0: modality id
        # 1: start frame index
        # 2: end frame index
        # 3: pid
        # 4: camera id
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            img_names = names[start_index-1:end_index]  
            img_paths = [osp.join(self.root, "Test", decoder_pic_path(img_name)) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths)) 

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class PRID2011(object):
   

    def __init__(self, root, min_seq_len=12, split_id=0, exclusive_sampling=True):

        """
        The splits are from the json file, following "https://github.com/KaiyangZhou/deep-person-reid/blob/master/
        torchreid/data/datasets/video/prid2011.py".
        For test, the identity set for RGB and Event modalities is the same. Therefore, we extract all the event/RGB
        tracklets and use them as the query and gallery sets, separately.
        """
        self.root = root
        split_path = os.path.join(self.root, 'splits_prid2011.json')
        splits = read_json(split_path)[split_id]
        train_dirs, test_dirs = splits['train'], splits['test']
        event_dir = os.path.join(self.root, 'prid_event/multi_shot')
        rgb_dir = os.path.join(self.root, 'prid_rgb/multi_shot')
        cams_path = ['cam_a', 'cam_b']
        # load training set
        train_tracklets_event, train_num_tracklets_event, train_num_imgs_per_tracklets_event, train_tracklets_rgb, train_num_tracklets_rgb, \
        train_num_imgs_per_tracklets_rgb, train_num_pids, train_event_labels, train_rgb_labels = self._process_data(train_dirs,
                                                                                            event_dir, rgb_dir,
                                                                                            cams_path=cams_path,
                                                                                            relabel=True,
                                                                                            min_seq_len=min_seq_len)
        # load test set
        test_tracklets_event, test_num_tracklets_event, test_num_imgs_per_tracklets_event, test_tracklets_rgb, test_num_tracklets_rgb, \
        test_num_imgs_per_tracklets_rgb, test_num_pids, test_event_labels, test_rgb_labels = self._process_data(test_dirs,
                                                                                            event_dir, rgb_dir,
                                                                                            cams_path=cams_path,
                                                                                            relabel=False,
                                                                                            min_seq_len=min_seq_len)

        # ---------------------------------------
        print("=> PRID2011 loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_event  | {:5d} | {:8d}".format(train_num_pids, train_num_tracklets_event))
        print("train_rgb   | {:5d} | {:8d}".format(train_num_pids, train_num_tracklets_rgb))
        print("test_event  | {:5d} | {:8d}".format(test_num_pids, test_num_tracklets_event))
        print("test_rgb   | {:5d} | {:8d}".format(test_num_pids, test_num_tracklets_rgb))
        print("---------------------------------")
        # exit()
        self.train_event = train_tracklets_event
        # print(f'train event vis: {self.train_event[:1]}')
        self.train_rgb = train_tracklets_rgb
        # print(f'train rgb vis: {self.train_rgb[:1]}')
        self.event_label = train_event_labels
        # print(f'even label: {self.event_label}')
        self.rgb_label = train_rgb_labels
        # print(f'rgb label: {self.rgb_label}')

        self.test_event = test_tracklets_event
        # print(f'test event vis: {self.test_event[:1]}')
        self.test_rgb = test_tracklets_rgb
        # print(f'test rgb vis: {self.test_rgb[:1]}')

        self.num_train_pids = train_num_pids
        # print(f'num train pids: {self.num_train_pids}')
        self.num_test_pids = test_num_pids
        # print(f'num test pids: {self.num_test_pids}')
        self.num_train_event_tracklets = train_num_tracklets_event
        # print(f'num train event tracklets: {self.num_train_event_tracklets}')
        self.num_train_rgb_tracklets = train_num_tracklets_rgb
        # print(f'num train rgb tracklets: {self.num_train_rgb_tracklets}')
        self.num_test_event_tracklets = test_num_tracklets_event
        # print(f'num test event tracklets: {self.num_test_event_tracklets}')
        self.num_test_rgb_tracklets = test_num_tracklets_rgb
        # print(f'num test rgb tracklets: {self.num_test_rgb_tracklets}')
        # exit()

    def _process_data(self, train_dirs, event_dir, rgb_dir, cams_path, relabel=False, min_seq_len=0):
        """
        return:
        tracklets list consisting of [((img_path1, img_path2, ...), pid, cam_id)]
        num_tracklets
        num_imgs_per_tracklets
        num_pids
        label
        """
        # build relabel mapping
        tracklets_rgb, tracklets_event = [], []
        rgb_labels, event_labels = [], []
        num_imgs_per_tracklets_rgb, num_imgs_per_tracklets_event = [], []
        relabel_mapping = dict()
        num_pids = len(train_dirs)
        for i, id in enumerate(train_dirs):
            relabel_mapping[id] = i
        # print(f'relabel mapping: {relabel_mapping}')
        # process rgb data
        for _, cam_path in enumerate(cams_path):
            rgb_with_cam_dir = os.path.join(rgb_dir, cam_path)
            event_with_cam_dir = os.path.join(event_dir, cam_path)
            for train_id in train_dirs:
                if relabel:
                    pid = relabel_mapping[train_id]
                else:
                    pid = int(train_id[-4:]) # "person_0001"
                cam_id = cam_path

                # load rgb tracklets
                train_id_rgb_dir = os.path.join(rgb_with_cam_dir, train_id)
                rgb_img_paths = [os.path.join(train_id_rgb_dir, i) for i in os.listdir(train_id_rgb_dir)]
                if len(rgb_img_paths) >= min_seq_len:
                    tracklets_rgb.append((rgb_img_paths, pid, cam_id))
                    num_imgs_per_tracklets_rgb.append(len(rgb_img_paths))
                    rgb_labels.append(pid)

                # load event tracklets
                train_id_event_dir = os.path.join(event_with_cam_dir, train_id)
                event_img_paths = [os.path.join(train_id_event_dir, i) for i in os.listdir(train_id_event_dir)]
                if len(event_img_paths) >= min_seq_len:
                    tracklets_event.append((event_img_paths, pid, cam_id))
                    num_imgs_per_tracklets_event.append(len(event_img_paths))
                    event_labels.append(pid)

        return tracklets_event, len(tracklets_event), num_imgs_per_tracklets_event, tracklets_rgb, len(tracklets_rgb), num_imgs_per_tracklets_rgb, num_pids, event_labels, rgb_labels

    def _process_data_test(self, train_dirs, event_dir, rgb_dir, cams_path, relabel=False, min_seq_len=0, exclusive_sampling=True):
        """
        each id has two tracklets from two cameras
        return:
        tracklets list consisting of [((img_path1, img_path2, ...), pid, cam_id)]
        num_tracklets
        num_imgs_per_tracklets
        num_pids
        label
        """
        # build relabel mapping
        tracklets_rgb, tracklets_event = [], []
        rgb_labels, event_labels = [], []
        num_imgs_per_tracklets_rgb, num_imgs_per_tracklets_event = [], []
        relabel_mapping = dict()
        num_pids = len(train_dirs)
        for i, id in enumerate(train_dirs):
            relabel_mapping[id] = i
        # print(f'relabel mapping: {relabel_mapping}')
        # process rgb data
        for _, cam_path in enumerate(cams_path):
            rgb_with_cam_dir = os.path.join(rgb_dir, cam_path)
            event_with_cam_dir = os.path.join(event_dir, cam_path)
            for train_id in train_dirs:
                if relabel:
                    pid = relabel_mapping[train_id]
                else:
                    pid = int(train_id[-4:]) # "person_0001"
                cam_id = cam_path

                # load rgb tracklets path
                train_id_rgb_dir = os.path.join(rgb_with_cam_dir, train_id)
                rgb_img_paths = [os.path.join(train_id_rgb_dir, i) for i in sorted(os.listdir(train_id_rgb_dir))]
                # load event tracklets path
                train_id_event_dir = os.path.join(event_with_cam_dir, train_id)
                event_img_paths = [os.path.join(train_id_event_dir, i) for i in sorted(os.listdir(train_id_event_dir))]

                if len(rgb_img_paths) >= min_seq_len:
                    tracklets_rgb.append((rgb_img_paths, pid, cam_id))
                    num_imgs_per_tracklets_rgb.append(len(rgb_img_paths))
                    rgb_labels.append(pid)
                if len(event_img_paths) >= min_seq_len:
                    tracklets_event.append((event_img_paths, pid, cam_id))
                    num_imgs_per_tracklets_event.append(len(event_img_paths))
                    event_labels.append(pid)

        return tracklets_event, len(tracklets_event), num_imgs_per_tracklets_event, tracklets_rgb, len(tracklets_rgb), num_imgs_per_tracklets_rgb, num_pids, event_labels, rgb_labels


class EventRgbDataManager(object):

    def __init__(self, root, min_seq_len=12, train_relabel=True, exclusive_sampling=True, train_split='train.json',
                 test_split='test.json'):
        self.root = root
        train_split_path = os.path.join(self.root, train_split)
        test_split_path = os.path.join(self.root, test_split)

        train_tracklets_event, train_num_tracklets_event, train_num_imgs_per_tracklets_event, train_tracklets_rgb, train_num_tracklets_rgb, \
        train_num_imgs_per_tracklets_rgb, train_num_pids, train_event_labels, train_rgb_labels = self._process_data(
            train_split_path,
            relabel=train_relabel,
            min_seq_len=min_seq_len)
        # load test set
        test_tracklets_event, test_num_tracklets_event, test_num_imgs_per_tracklets_event, test_tracklets_rgb, test_num_tracklets_rgb, \
        test_num_imgs_per_tracklets_rgb, test_num_pids, test_event_labels, test_rgb_labels = self._process_data(
            test_split_path,
            relabel=False,
            min_seq_len=min_seq_len)
        print(f'check test data: ')
        # print(f'rgb tacklets: \n  {test_tracklets_rgb[:10]}')
        # print(f'event tacklets: \n  {test_tracklets_event[:10]}')
        # ---------------------------------------
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_event  | {:5d} | {:8d}".format(train_num_pids, train_num_tracklets_event))
        print("train_rgb   | {:5d} | {:8d}".format(train_num_pids, train_num_tracklets_rgb))
        print("test_event  | {:5d} | {:8d}".format(test_num_pids, test_num_tracklets_event))
        print("test_rgb   | {:5d} | {:8d}".format(test_num_pids, test_num_tracklets_rgb))
        print("---------------------------------")
        # exit()
        self.train_event = train_tracklets_event
        # print(f'train event vis: {self.train_event[:1]}')
        self.train_rgb = train_tracklets_rgb
        # print(f'train rgb vis: {self.train_rgb[:1]}')
        self.event_label = train_event_labels
        # print(f'even label: {self.event_label}')
        self.rgb_label = train_rgb_labels
        # print(f'rgb label: {self.rgb_label}')

        self.test_event = test_tracklets_event
        # print(f'test event vis: {self.test_event[:1]}')
        self.test_rgb = test_tracklets_rgb
        # print(f'test rgb vis: {self.test_rgb[:1]}')

        self.num_train_pids = train_num_pids
        # print(f'num train pids: {self.num_train_pids}')
        self.num_test_pids = test_num_pids
        # print(f'num test pids: {self.num_test_pids}')
        self.num_train_event_tracklets = train_num_tracklets_event
        # print(f'num train event tracklets: {self.num_train_event_tracklets}')
        self.num_train_rgb_tracklets = train_num_tracklets_rgb
        # print(f'num train rgb tracklets: {self.num_train_rgb_tracklets}')
        self.num_test_event_tracklets = test_num_tracklets_event
        # print(f'num test event tracklets: {self.num_test_event_tracklets}')
        self.num_test_rgb_tracklets = test_num_tracklets_rgb
        # print(f'num test rgb tracklets: {self.num_test_rgb_tracklets}')
        # exit()

    def _process_data(self, split_path, relabel=False, min_seq_len=0):
        """
        return:
        tracklets list consisting of [((img_path1, img_path2, ...), pid, cam_id)]
        num_tracklets
        num_imgs_per_tracklets
        num_pids
        label
        """
        # build relabel mapping
        tracklets_rgb, tracklets_event = [], []
        rgb_labels, event_labels = [], []
        num_imgs_per_tracklets_rgb, num_imgs_per_tracklets_event = [], []
        relabel_mapping = dict()
        with open(split_path, 'r') as f:
            data = json.load(f)
        num_pids = len(data.keys())
        print(f'num pids: {num_pids}')
        for i, id in enumerate(list(data.keys())):
            relabel_mapping[id] = i
        # print(f'relabel mapping: {relabel_mapping}')
        # process rgb data
        for id in list(data.keys()):
            if relabel:
                pid = relabel_mapping[id]
            else:
                pid = int(id)
            event_cam_list, rgb_cam_list = data[id]['event_cam'], data[id]['rgb_cam']
            for cam_id in data[id].keys():
                if cam_id == 'event_cam' or cam_id == 'rgb_cam':
                    continue
                tracklets = data[id][cam_id]
                for track in tracklets:
                    track_rgb, track_event = track[0], track[1]
                    if cam_id in event_cam_list:
                        if len(track_event) >= min_seq_len:
                            tracklets_event.append((track_event, pid, cam_id))
                            num_imgs_per_tracklets_event.append(len(track_event))
                            event_labels.append(pid)
                    if cam_id in rgb_cam_list:
                        if len(track_rgb) >= min_seq_len:
                            tracklets_rgb.append((track_rgb, pid, cam_id))
                            num_imgs_per_tracklets_rgb.append(len(track_rgb))
                            rgb_labels.append(pid)
        return tracklets_event, len(tracklets_event), num_imgs_per_tracklets_event, tracklets_rgb, len(
            tracklets_rgb), num_imgs_per_tracklets_rgb, num_pids, event_labels, rgb_labels


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


if __name__ == '__main__':
    dataset = VCM()
    