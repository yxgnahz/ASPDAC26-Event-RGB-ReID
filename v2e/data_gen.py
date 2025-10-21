import os
import re
import sys
import glob
from multiprocessing.pool import Pool
from multiprocessing import Manager
import subprocess

import numpy as np
from PIL import Image


def HITSZ_transform(person_folder):
    if not os.path.exists(person_folder):
        print(os.path.exists(person_folder), person_folder)
        return

    img_folders = glob.glob(os.path.join(person_folder, 'D*'))
    for img_folder in img_folders:
        #resized first
        resized_img_folder = img_folder.replace('rgb', 'rgb_resized')
        if not os.path.exists(resized_img_folder):
            os.makedirs(resized_img_folder)
        img_names = glob.glob(os.path.join(img_folder, '*.jpg'))
        h = 288
        w = 144
        for img_name in img_names:
            idx = re.findall(r'\d+', img_name)[-1]
            img = Image.open(img_name)
            img_resized = img.resize((w,h))
            img_resized.save(resized_img_folder + '/%04d.jpg'%int(idx))
        
        #using v2e.py to get .txt file
        event_folder = img_folder.replace('rgb', 'event')
        npy_folder = img_folder.replace('rgb', 'npy')
        if not os.path.exists(event_folder):
            os.makedirs(event_folder)
        if not os.path.exists(npy_folder):
            os.makedirs(npy_folder)
        cmd = "v2e.py \
            -o " + event_folder + \
            " --overwrite --skip_video_output --disable_slomo \
            --input " + resized_img_folder + \
            " --input_frame_rate 5 --dvs_h5 DVS_H5 --dvs_text DVS_TEXT"
        subprocess.run(cmd)

        
        #generate .png images
        with open(event_folder + '/DVS_TEXT.txt', 'r') as fp:
            lines = fp.readlines()
            lines = [line.strip() for line in lines[6:]]
        event = np.zeros((h,w))
        event_frame = np.zeros((h,w,2),dtype=np.uint8)
        event_frame_virtual = np.ones((h,w,3), dtype=np.uint8) * 255
        time_interval = 0.2
        end_time = 0.2
        idx = 6
        for line in lines:
            t = line.split()[0]
            if float(format(float(t), '.1f')) >= end_time:
                end_time += time_interval
                i=0
                while i<h:
                    j=0
                    while j<w:
                        if event[i][j] > 0:
                            event_frame_virtual[i][j][1] = 0
                            event_frame_virtual[i][j][2] = 0
                        if event[i][j] < 0:
                            event_frame_virtual[i][j][0] = 0
                            event_frame_virtual[i][j][1] = 0
                        j+=1
                    i+=1

                savepath_img = event_folder + '/%04d.png'%idx
                im = Image.fromarray(event_frame_virtual, mode="RGB")
                im.save(savepath_img)
                savepath_npy = npy_folder + '/%04d.npy'%idx
                np.save(savepath_npy, event_frame)
                idx += 5
                event = np.zeros((h,w))
                event_frame = np.zeros((h,w,2),dtype=np.uint8)
                event_frame_virtual = np.ones((h,w,3), dtype=np.uint8) * 255
            x = int(line.split()[1])
            y = int(line.split()[2])
            p = int(line.split()[3])
            #print(x, y, p ,t)
            if p == 0:
                p = -1
                event_frame[y][x][0] += 1
            else:
                event_frame[y][x][1] += 1
            event[y][x] += p
        
    return person_folder


def Mars_generate_DVS(tracklet_folder):
    if len(os.listdir(tracklet_folder)) < 1:#no masked picture
        return tracklet_folder
    h = 256
    w = 128
    #using v2e.py to get .txt file
    event_folder = tracklet_folder.replace('rgb', 'event')
    npy_folder = tracklet_folder.replace('rgb', 'npy')
    if not os.path.exists(event_folder):
        os.makedirs(event_folder)
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)
    cmd = ["v2e.py",
        "-o", event_folder,
        "--overwrite", 
        "--skip_video_output", 
        "--disable_slomo",
        "--input", tracklet_folder,
        "--input_frame_rate", "1", 
        "--dvs_text", "DVS_TEXT"]
    subprocess.run(cmd)
    
    #generate .png images
    with open(event_folder + '/DVS_TEXT.txt', 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines[6:]]
    event = np.zeros((h,w))
    event_frame = np.zeros((h,w,2),dtype=np.uint8)
    event_frame_virtual = np.ones((h,w,3), dtype=np.uint8) * 255
    time_interval = 1
    end_time = 1
    idx = 2
    #tracklet_statistics = {'C1Per':0,'C2Per':0,'C3Per':0,'C4Per':0,'C5Per':0,'C6Per':0,
    #            'MaxT':0,'MinT':10000,'Max0':0,'Max1':0}
    for line in lines:
        t = line.split()[0]
        if float(format(float(t), '.0f')) >= end_time:
            end_time += time_interval
            i=0
            while i<h:
                j=0
                while j<w:
                    if event[i][j] > 0:
                        event_frame_virtual[i][j][1] = 0
                        event_frame_virtual[i][j][2] = 0
                    if event[i][j] < 0:
                        event_frame_virtual[i][j][0] = 0
                        event_frame_virtual[i][j][1] = 0
                    j+=1
                i+=1

            savepath_img = event_folder + '/F%03d.png'%idx
            im = Image.fromarray(event_frame_virtual, mode="RGB")
            im.save(savepath_img)
            savepath_npy = npy_folder + '/F%03d.npy'%idx
            np.save(savepath_npy, event_frame)
            idx += 1
            event = np.zeros((h,w))
            event_frame = np.zeros((h,w,2), dtype=np.uint8)
            event_frame_virtual = np.ones((h,w,3), dtype=np.uint8) * 255
        x = int(line.split()[1])
        y = int(line.split()[2])
        p = int(line.split()[3])
        #print(x, y, p ,t)
        if p == 0:
            p = -1
            event_frame[y][x][0] += 1
        else:
            event_frame[y][x][1] += 1
        event[y][x] += p

    return tracklet_folder


def Mars_transform(person_folder):
    if not os.path.exists(person_folder):
        print(os.path.exists(person_folder), person_folder)
        return
    cam_folders = glob.glob(os.path.join(person_folder, 'C*'))
    for cam_folder in cam_folders:
        tracklet_folders = glob.glob(os.path.join(cam_folder, 'T*'))
        with Pool(4) as pool:
            for tracklet_folder in pool.imap_unordered(Mars_generate_DVS, tracklet_folders):
                print("finished",  tracklet_folder)
        

if __name__ == '__main__':
    task = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

    # handle vcm
    #HITSZ_folder = '/mnt/proj202/yurui/yurui/prev_work/data/HITSZ/' + task
    #person_folders = [HITSZ_folder + '/%04d/rgb_resized'%i for i in range(start, end+1)]
    #with Pool(48) as pool:
    #    for person_folder in pool.imap_unordered(HITSZ_transform, person_folders):
    #        print("finished", person_folder)

    # handle mars
    Mars_folder = '/mnt/proj202/yurui/yurui/prev_work/data/mars/bbox_' + task
    person_folders = [Mars_folder + '/%04d/masked_rgb_mask2former_max_area_ESR_4'%i for i in range(start, end+1)]
    
    for person_folder in person_folders:
        Mars_transform(person_folder)