# MMflood-segmentation project
This repository serves as the code write-up for ENVIR ST 900 – AI for Earth Observation at the University of Wisconsin–Madison.

If you would like to explore the models I trained and the visualization examples 
(including data visualizations, training-loss curves, and evaluation metrics), 
please see the **[envir900_project/](envir900_project/)** folder.  

A detailed project reflection is provided in **[Reflection.md](envir900_project/Reflection.md)** in that folder.

## Project Overview

Floods are among the most widespread and damaging natural disasters. After a major flood, emergency managers need fast and reliable maps of where the water actually is. Optical satellite images are often blocked by clouds and darkness, so they are not always available when floods happen.

This project explores whether deep learning models can map flooded areas from Sentinel-1 radar imagery (SAR), which works day and night and can see through clouds. I use an open dataset called **MMFlood**, which collects flood events from many countries, and I train models to automatically segment (outline) flooded pixels in each image.

However, most images in MMFlood contain very little water, which have often less than 5% of the pixels are flooded. My project focuses on dealing with this imbalance, testing different model architectures, and evaluating how design choices (such as backbones, sampling strategies, and extra inputs like DEM) affect flood-mapping performance.

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

### Source  
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
### Data Modalities  

| Image    | Description                                           | Format            | Bands        |
| -------- | ----------------------------------------------------- | ----------------- | ------------ |
| S1 raw   | Georeferenced Sentinel-1 imagery, IW GRD              | GeoTIFF Float32   | 0: VV, 1: VH |
| DEM      | MapZen Digital Elevation Model                        | GeoTIFF Float32   | 0: elevation |
| Hydrogr. | Binary map of permanent water basins, OSM             | GeoTIFF Uint8     | 0: hydro     |
| Mask     | Manually validated ground truth label, Copernicus EMS | GeoTIFF Uint8     | 0: gt        |

## Code and installation

To run this project, please set up the environment using the packages listed in 
[requirements.txt](requirements.txt),Python 3.8+ recommended.
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

## Results & Key Findings
Here I summarize the main quantitative and qualitative findings.  

| Model             | Channels | Weighted Sampling | Mask Ratio | Precision | Recall  |  IoU   |   F1   |
|------------------|----------|-------------------|------------|-----------|---------|--------|--------|
| U-Net (R18)      | SAR      | ✓                 | ✓          | 0.6564    | **0.9159** | 0.6191 | 0.7647 |
| U-Net (R50)      | SAR      | ✗                 | ✗          | 0.5639    | 0.1651  | 0.0790 | 0.1464 |
| U-Net (R50)      | SAR      | ✓                 | ✓          | 0.7516    | 0.8388  | **0.6568** | **0.7928** |
| U-Net (R101)     | SAR      | ✓                 | ✓          | 0.7376    | 0.8055  | 0.6261 | 0.7700 |
| U-Net (R50)      | SAR+DEM  | ✓                 | ✓          | **0.7633** | 0.8150  | 0.6506 | 0.7883 |
| DeepLabv3+ (R50) | SAR      | ✓                 | ✓          | 0.6760    | 0.9028  | 0.6302 | 0.7731 |

### Key Findings

**1. Class imbalance handling is essential.**  
Models trained without weighted sampling and without mask-ratio filtering (U-Net R50 baseline) produced almost empty flood predictions, with recall of 0.1651 and IoU of 0.0790.  
After applying weighted sampling and a 2% mask-ratio threshold, recall increased to 0.8388 and IoU to 0.6568, demonstrating that imbalance handling is critical for MMFlood.

**2. Medium-depth backbones perform best.**  
ResNet101 did not outperform ResNet50. The deeper backbone tended to overfit background noise, while the ResNet50 backbone achieved the strongest results (IoU = 0.6568, F1 = 0.7928).  
This suggests that medium-depth encoders capture SAR features more effectively in this dataset.

**3. DEM does not improve performance when naively fused.**  
Stacking DEM as an additional channel provided no clear benefit. The SAR+DEM model had slightly higher precision but no improvement in IoU or F1.  
This indicates that DEM requires a more explicit fusion strategy (e.g., a separate encoder) rather than simple channel concatenation.

**4. DeepLabv3+ performs well but does not surpass U-Net.**  
DeepLabv3+ (R50) achieved IoU = 0.6302 and F1 = 0.7731.  
Although it benefits from multi-scale context, it did not outperform U-Net with ResNet50 on this dataset, likely due to dataset size and noise characteristics.

**5. U-Net with ResNet18 shows very high recall but lower IoU.**  
U-Net (R18) achieved the highest recall (0.9159) but lower IoU (0.6191), suggesting a tendency to over-predict flooded regions.  
Shallow encoders may generalize more aggressively but sacrifice precision.

Overall, the best-performing model in this study was **U-Net with a ResNet50 backbone**, trained with **weighted sampling** and **mask-ratio filtering**, using **SAR-only** inputs.


## Acknowledgment
This project uses the MMFlood dataset and partially adopts components from the original codebase:
[https://github.com/floods-mm/floods](https://github.com/edornd)
We thank the authors for making these resources publicly available.
The implementation was adapted for course requirements, and may differ from the official repository in configuration structure and training scripts.
