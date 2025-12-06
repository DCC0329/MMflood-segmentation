# MMflood-segmentation project
This repository serves as the code write-up for ENVIR ST 900 – AI for Earth Observation at the University of Wisconsin–Madison.

If you would like to explore the models I trained and the visualization examples 
(including data visualizations, training-loss curves, and evaluation metrics), 
please see the **[envir900_project/](envir900_project/)** folder.  

A detailed project reflection is provided in **[Reflection.md](envir900_project/Reflection.md)** in that folder.

##  Summary  
This project develops deep learning models to detect flooded areas from Sentinel-1 SAR imagery.
I compare U-Net and DeepLabv3+ architectures, evaluate different backbones (ResNet18/50/101),
test imbalance-handling strategies (weighted sampling, mask_ratio), and explore multimodal input (SAR+DEM).

## Research Question & Motivation

### Research Question:

Can deep learning models accurately map flood extents from noisy, highly imbalanced SAR data?
How do architecture choices, backbones, and sampling strategies affect performance?

### Motivation:
Floods cause widespread damage, and floods are among the most destructive natural hazards.  
Accurate large-scale flood mapping enables rapid response and risk assessment, but traditional optical remote sensing is limited by clouds.
SAR works in all weather conditions, making it ideal for rapid flood mapping.

However, SAR datasets are noisy and extremely imbalanced (most tiles contain <5% flood pixels),
making segmentation challenging. This project investigates modeling approaches to improve performance.

##  Dataset  
from Zenodo: https://zenodo.org/record/6534637
  <img width="963" height="436" alt="image" src="https://github.com/user-attachments/assets/20746669-6b78-4f00-92f4-24cfc61cc1d6" />

### Structure
The dataset is organized in directories, with a JSON file providing metadata and other information such as the split configuration we selected.
Its internal structure is as follows:

```
activations/
├─ EMSR107-1/
├─ .../
├─ EMSR548-0/
│  ├─ DEM/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
│  ├─ hydro/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
│  ├─ mask/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
│  ├─ s1_raw/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
activations.json
```
| Image    | Description                                           | Format            | Bands        |
| -------- | ----------------------------------------------------- | ----------------- | ------------ |
| S1 raw   | Georeferenced Sentinel-1 imagery, IW GRD              | GeoTIFF Float32   | 0: VV, 1: VH |
| DEM      | MapZen Digital Elevation Model                        | GeoTIFF Float32   | 0: elevation |
| Hydrogr. | Binary map of permanent water basins, OSM             | GeoTIFF Uint8     | 0: hydro     |
| Mask     | Manually validated ground truth label, Copernicus EMS | GeoTIFF Uint8     | 0: gt        |

## Code and installation

To run this project, please set up the environment using the packages listed in 
[requirements.txt](requirements.txt),
and you need to clone it into a directory of choice and create a python environment.
```bash
git clone git@github.com:edornd/mmflood.git && cd mmflood
python3 -m venv .venv
pip install -r requirements.txt
```

Everything goes through the `run` command.
Run `python run.py --help` for more information about commands and their arguments.


### Data preparation
To prepare the raw data by tiling and preprocessing, you can run:
`python run.py prepare --data-source [PATH_TO_ACTIVATIONS] --data-processed [DESTINATION]`


### Training
Training uses HuggingFace `accelerate` to provide single-gpu and multi-gpu support.
To launch on a single GPU:
```bash
CUDA_VISIBLE_DEVICES=... python run.py train [ARGS]
```

### Testing
Testing is run on non-tiled images (the preprocessing will format them without tiling).
You can run the test on a single GPU using the `test` command.
At the very least, you need to point the script to the output directory.
If no checkpoint is provided, the best one (according to the monitored metric) will be selected automatically.
You can also avoid storing outputs with `--no-store-predictions`.
```bash
CUDA_VISIBLE_DEVICES=... python run.py test --data-root [PATH_TO_OUTPUT_DIR] [--checkpoint-path [PATH]]
```
## Acknowledgment
This project uses the MMFlood dataset and partially adopts components from the original codebase:
[https://github.com/floods-mm/floods](https://github.com/edornd)
We thank the authors for making these resources publicly available.
