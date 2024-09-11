# Counter-forensic-attack-detection
### Dataset extraction from parquet files-done, instead point to parquet files
### other all other images under data folder-instead point to google drive
### image reconstruction using auto-encoder-done
### extracting CLIP encodings
### CLIP classifier training
### DCT classifier training for binary and multiclass
### trained models
### attack detection (ensemble model)
### testing dct on resized images

Link to dataset--https://drive.google.com/drive/folders/1XIhoGIguyhMEaedWKkGtzVGzzXjXuEEt?usp=sharing
# ProjectName

This repository contains the code and documentation for **ProjectName**, which aims to solve [brief description of the problem/project].

## Overview

**ProjectName** aims to [short summary of your project]. The project provides [key features, tools, or analysis it offers].

## Directory Structure
Please ensure the test data is stored in the following hierarchy
```bash
ICVGIP_Addressing_DM/
│
├──
├── CLIP/                    # Source code files
├──  DCT_Classification/                     # Data storage
│   ├── Models/              # ML/DL models
│   ├── Testing3classDCT.py              # Main executable script
├── Data/               # Contains zip files for dataset, csv files
├── Ensemble Model/                           # Contains code to test ensemble model 
│  ├── test_ensemble_model.py              
└── README.md                # Project documentation
```
## Test Code
To run the code, the following dependencies need to be installed:
```
*numpy
*scikit-image
*scikit-learn
*tensorflow
*scipy
*matplotlib
*Pillow
*torch
*pandas
```

### Script
The `file.py` script requires...
The test can be executed as follows:
```
code
```


# 9/9/24
### Done-3dct models, 3 way clip model, 3 way dct
### Left-code to test everything, 3way dct cleanup, all bin dct cleanup
