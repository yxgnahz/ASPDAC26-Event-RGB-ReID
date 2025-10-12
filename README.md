# Event-RGB-ReID

## Data Processing

### PRID2011

The PRID2011 dataset used in this project is sourced from the [SDCL repository by Chengzhi Cao](https://github.com/Chengzhi-Cao/SDCL).

After downloading the raw data, we perform filtering and split the remaining identities into training and testing subsets. The lists of identities for each split are provided in `data/PRID2011/train.json` and `data/PRID2011/test.json`.

Once prepared, organize the directory structure to match the following layout:

```
data/
└── PRID2011/
    ├── prid_rgb/
    ├── prid_event/
    ├── train.json
    └── test.json
```

### MARS

The MARS dataset is obtained from the [official evaluation repository](https://github.com/liangzheng06/MARS-evaluation) and should be downloaded into `data/mars`.

Following the official processing script, the RGB frames for each tracklet are stored under a folder hierarchy such as `data/mars/bbox_train/0001/C1/T0001/F001.jpg`. Create an additional `rgb` folder for every identity to hold the raw RGB images, resulting in paths like `data/mars/bbox_train/0001/rgb/C1/T0001/F001.jpg`.

Convert the RGB frames to the event-image representation with [v2e](https://github.com/SensorsINI/v2e) using the default parameter settings. Save the converted frames in a sibling `event` directory that mirrors the RGB structure, for example `data/mars/bbox_train/0001/event/C1/T0001/F001.png`.

The predefined training and testing splits are provided in `data/mars/train.json` and `data/mars/test.json`.

The prepared directory should resemble the following structure:

```
data/
└── mars/
    ├── bbox_train/
    │   └── ...
    │       ├── event/
    │       └── rgb/
    ├── bbox_test/
    │   └── ...
    ├── info/
    ├── train.json
    └── test.json
```
