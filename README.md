# Counter-forensic-attack-detection


Link to dataset--https://drive.google.com/drive/folders/1XIhoGIguyhMEaedWKkGtzVGzzXjXuEEt?usp=sharing

This repository contains the code and documentation for **ProjectName**, which aims to solve [brief description of the problem/project].

## Overview

**ProjectName** aims to [short summary of your project]. The project provides [key features, tools, or analysis it offers].

## Directory Structure
Please ensure the test data is stored in the following hierarchy
```bash
ICVGIP_Addressing_DM/
│
├──
├── CLIP/                     # Contains code to test CLIP model
│   ├── Test_3class_CLIP.py   # Script to test three class CLIP
├──  DCT_Classification/      # Folder containing DCT models/code
│   ├── Models/               # Folder containing models
│   ├── Test_3class_DCT.py    # Script to test 3 Class DCT
│   ├── Test_binary_DCT.py    # Script to test Binary DCT
├── Data/                     # Contains zip files for dataset, CLIP encodings
│   ├── CLIP_encodings_test/  # Folder containing CLIP encodings
│   ├── COCO_test/            # Folder containing COCO test images
│   ├── ucid/                 # Folder containing UCID test images
│   ├── imagenet/             # Folder containing Imagenet test images
│   ├── deep_floyd/           # Folder containing DeepFLoyd test images
│   ├── sd14/                 # Folder containing StableDiffusion 1.4 test images
│   ├── sd21/                 # Folder containing StableDiffusion 2.1 test images
│   ├── sdxl/                 # Folder containing StableDiffusion XL test images
│   ├── test/                 # Folder containing DM reconstructed test images
├── Ensemble Model/           # Contains code to test ensemble model 
│  ├── Test_ensemble_model.py # Script to test ensemble model             
└── README.md                # Project documentation
```
## Test Code
To run the code, the following dependencies need to be installed:
```
* numpy
* scikit-image
* scikit-learn
* tensorflow
* scipy
* matplotlib
* Pillow
* torch
* pandas
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
