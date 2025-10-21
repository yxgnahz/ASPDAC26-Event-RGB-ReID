import os
import re
import glob
import subprocess
from typing import Literal
from pathlib import Path
from multiprocessing.pool import Pool

import numpy as np
from PIL import Image

root_path = Path(__file__).parent.parent


#generate from DVS_txt
def generate_hitsz_person(person_folder, h=288, w=144, time_interval=0.2, start_time=0.2, idx_interval=5):
    """
    Handling person of the MARS dataset

    Params:
        person_folder (str): Path to person's folder, containing subfolders named 'D*'.
        h (int): Frame height after resize.
        w (int): Frame width after resize.
        time_interval (float): Time interval (in seconds) between frames.
        start_time (float): Initial time to start saving frames.
        idx_interval (int): Frame index increment between saved frames.
    """
    # no such person id
    if not os.path.exists(person_folder):
        print(os.path.exists(person_folder), person_folder)
        return

    for img_folder in glob.glob(os.path.join(person_folder, 'D*')):
        # resize images to a uniform size
        resized_img_folder = img_folder.replace('rgb', 'rgb_resized')
        os.makedirs(resized_img_folder, exist_ok=True)
        for img_path in glob.glob(os.path.join(img_folder, '*.jpg')):
            idx = int(re.findall(r'\d+', img_path)[-1])
            img = Image.open(img_path)
            img.resize((w, h)).save(os.path.join(resized_img_folder, f"{idx:04d}.jpg"))
        
        # use v2e to get DVS file
        event_folder = img_folder.replace('rgb', 'event')
        npy_folder = img_folder.replace('rgb', 'npy')
        os.makedirs(event_folder, exist_ok=True)
        os.makedirs(npy_folder, exist_ok=True)
        cmd = "v2e.py \
            -o " + event_folder + \
            " --overwrite --skip_video_output --disable_slomo \
            --input " + resized_img_folder + \
            " --input_frame_rate 5 --dvs_h5 DVS_H5 --dvs_text DVS_TEXT"
        subprocess.run(cmd, check=True)
        
        # generate png images
        with open(event_folder + '/DVS_TEXT.txt', 'r') as fp:
            lines = [line.strip() for line in fp.readlines()[6:]]
        event = np.zeros((h,w))
        event_frame = np.zeros((h,w,2),dtype=np.uint8)
        event_frame_virtual = np.ones((h,w,3), dtype=np.uint8) * 255
        idx = 1 + idx_interval
        cur_end_time = start_time + time_interval
        for line in lines:
            t, x, y, p = line.split()[:4]
            t, x, y, p = float(t), int(x), int(y), int(p)
            p = 1 if p > 0 else -1
            if t >= cur_end_time:
                cur_end_time += time_interval
                pos_mask = event > 0
                neg_mask = event < 0
                event_frame_virtual[pos_mask] = [255, 0, 0]   # Red for positive
                event_frame_virtual[neg_mask] = [0, 0, 255]   # Blue for negative

                savepath_img = os.path.join(event_folder, f"F{idx:04d}.png")
                Image.fromarray(event_frame_virtual, mode="RGB").save(savepath_img)
                savepath_npy = os.path.join(npy_folder, f"F{idx:04d}.npy")
                np.save(savepath_npy, event_frame)

                idx += idx_interval
                event.fill(0)
                event_frame.fill(0)
                event_frame_virtual.fill(255)
            event_frame[y, x, 0 if p < 0 else 1] += 1
            event[y, x] += p

    return person_folder


def run_hitsz(task: Literal['Train', 'Test'], start: int, end: int, workers: int = 48):
    """
    Batch process HITSZ dataset transformation

    params:
        task (str): dataset subfolder, e.g., 'train', 'test', 'val'
        start (int): starting person index
        end (int): ending person index
        workers (int, optional): number of parallel processes (default: 48)
    """
    hitsz_folder = f"{root_path}/data/HITSZ/{task}"
    person_folders = [f"{hitsz_folder}/{i:04d}/rgb" for i in range(start, end + 1)]

    print(f"[HITSZ] Start processing {len(person_folders)} folders from {start} to {end}")
    with Pool(workers) as pool:
        for person_folder in pool.imap_unordered(generate_hitsz_person, person_folders):
            print(f"Finish processing {person_folder}")
    print("[HITSZ] All done ✅")
