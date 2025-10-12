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
