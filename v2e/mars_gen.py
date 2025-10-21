import os
import glob
import subprocess
from typing import Literal
from pathlib import Path
from multiprocessing.pool import Pool

import numpy as np
from PIL import Image

root_path = Path(__file__).parent.parent


def generate_mars_tracklet(tracklet_folder, h=256, w=128, time_interval=1, start_time=1, idx_interval=1):
    """
    Handling tracklet of the MARS dataset person

    Params:
        tracklet_folder (str): Path to tracklet's folder
        h (int): Frame height.
        w (int): Frame width.
        time_interval (float): Time interval (in seconds) between frames.
        start_time (float): Initial time to start saving frames.
        idx_interval (int): Frame index increment between saved frames.
    """
    # no pictures in this tracklet
    if len(os.listdir(tracklet_folder)) < 1:
        return tracklet_folder

    # use v2e to get DVS file
    event_folder = tracklet_folder.replace('rgb', 'event')
    npy_folder = tracklet_folder.replace('rgb', 'npy')
    os.makedirs(event_folder, exist_ok=True)
    os.makedirs(npy_folder, exist_ok=True)
    cmd = ["v2e.py",
        "-o", event_folder,
        "--overwrite", 
        "--skip_video_output", 
        "--disable_slomo",
        "--input", tracklet_folder,
        "--input_frame_rate", "1", 
        "--dvs_text", "DVS_TEXT"]
    subprocess.run(cmd, check=True)

    #generate png images
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

            savepath_img = os.path.join(event_folder, f"F{idx:03d}.png")
            Image.fromarray(event_frame_virtual, mode="RGB").save(savepath_img)
            savepath_npy = os.path.join(npy_folder, f"F{idx:03d}.npy")
            np.save(savepath_npy, event_frame)

            idx += idx_interval
            event.fill(0)
            event_frame.fill(0)
            event_frame_virtual.fill(255)
        event_frame[y, x, 0 if p < 0 else 1] += 1
        event[y, x] += p

    return tracklet_folder


def generate_mars_person(person_folder, workers: int = 4):
    """
    Handling person of the MARS dataset

    Params:
        person_folder (str): Path to person's folder
        workers (int, optional): number of parallel processes (default: 48)
    """
    if not os.path.exists(person_folder):
        print(f"{person_folder} not exist")
        return
    cam_folders = glob.glob(os.path.join(person_folder, 'C*'))
    for cam_folder in cam_folders:
        tracklet_folders = glob.glob(os.path.join(cam_folder, 'T*'))
        with Pool(workers) as pool:
            for tracklet_folder in pool.imap_unordered(generate_mars_tracklet, tracklet_folders):
                print(f"Finish processing {tracklet_folder}")


def run_mars(task: Literal['train', 'test'], start: int, end: int, workers: int = 4):
    """
    Batch process MARS dataset transformation

    params:
        task (str): dataset subfolder, e.g., 'train', 'test', 'val'
        start (int): starting person index
        end (int): ending person index
        workers (int, optional): number of parallel processes (default: 48)
    """
    mars_folder = f"{root_path}/data/mars/bbox_{task}"
    person_folders = [f"{mars_folder}/{i:04d}/rgb" for i in range(start, end + 1)]

    print(f"[Mars] Start processing {len(person_folders)} folders from {start} to {end}")
    for person_folder in person_folders:
        generate_mars_person(person_folder, workers)
        print(f"Finish processing {person_folder}")

    print("[Mars] All done ✅")
