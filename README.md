# Counter-forensic-attack-detection


Link to dataset--https://drive.google.com/drive/folders/1XIhoGIguyhMEaedWKkGtzVGzzXjXuEEt?usp=sharing

This is the official repository that contains the code and documentation for the paper 'Addressing Diffusion Model Based Counter-Forensic Image Manipulation for Synthetic Image Detection'.

## Overview

This work aims to find ways to detect counterforensic image manipulations in Synthetic (Diffusion model-based) images. This repository provides some of the code to test our model.

## Directory Structure
Please ensure the repository follows the following structure
```bash
ICVGIP_Addressing_DM/
│
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
├── Ensemble_Model/           # Contains code to test ensemble model 
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
The following commands can be used to obtain the test results presented in the paper.
To test the binary DCT based classifiers, use the following command. The parsable input 'real' can take the inputs 'COCO', 'UCID' or 'ImageNet', representing the corresponding real image source. 'Mode' can take the inputs 'rf', 'rm', 'fm', to decide the binary classifier to be tested, i.e, 'Real vs. fake', 'Real vs. Manipulated' or 'Fake vs. Manipulated'. 
```
python ICVGIP_Addressing_DM/DCT_Classification/Test_binary_DCT.py --real 'COCO' --mode 'rf'
```
To test the three class DCT based classifier, use the following command. The parsable input 'mode' can take the inputs 'COCO', 'UCID' or 'ImageNet', representing the corresponding real image source.
```
python ICVGIP_Addressing_DM/DCT_Classification/Test_3class_DCT.py --mode 'COCO'
```
To test the three class CLIP based classifier, use the following command. The parsable input 'mode' can take the inputs 'COCO', 'UCID' or 'ImageNet', representing the corresponding real image source.
```
python ICVGIP_Addressing_DM/CLIP/Test_3class_CLIP.py --mode 'COCO'
```
To test the ensemble model, use the following command. The parsable input 'mode' can take the inputs 'COCO', 'UCID' or 'ImageNet', representing the corresponding real image source.
```
python ICVGIP_Addressing_DM/Ensemble_Model/Test_ensemble_model.py --mode 'COCO'
```


