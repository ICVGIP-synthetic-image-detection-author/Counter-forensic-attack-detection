import numpy as np
import os
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Synthetic Image Detection with 3-class classifier.')
parser.add_argument('--mode', type=str, default='COCO', help='Real image set.(COCO (default), UCID, ImageNet)')

# Parse arguments
args = parser.parse_args()

# Load precomputed CLIP encodings from the CSV file (use default if not provided)
if args.mode=='COCO':
  train_data = pd.read_csv('../data/CLIP_encodings_test/threewayenc_testcoco200.csv')
elif args.mode=='UCID':
  train_data = pd.read_csv('../data/CLIP_encodings_test/threewayenc_testucid.csv')
elif args.mode=='ImageNet':
  train_data = pd.read_csv('../data/CLIP_encodings_test/threewayenc_testimagenet.csv')
# Split data based on the type
true_real_clip = train_data[train_data['class'] == 'real'].drop(columns=['class', 'filename']).to_numpy()
true_fake_clip = train_data[train_data['class'] == 'fake'].drop(columns=['class', 'filename']).to_numpy()
man_real_clip = train_data[train_data['class'] == 'man'].drop(columns=['class', 'filename']).to_numpy()

# Combine the data
X_clip = np.vstack((true_real_clip, true_fake_clip, man_real_clip))

# Define labels
y_real = np.ones(len(true_real_clip))
y_fake = np.zeros(len(true_fake_clip))
y_man = np.full(len(man_real_clip), 2)
y_clip = np.hstack((y_real, y_fake, y_man))

# Load models
three_class_clip_classifier = load_model('../CLIP/3wayclip.h5')

# Predict using the 3-class CLIP classifier
three_class_clip_prob = three_class_clip_classifier.predict(X_clip)
three_class_clip_out = np.argmax(three_class_clip_prob, axis=1)  # Get class predictions
three_class_clip_out_one_hot = tf.keras.utils.to_categorical(three_class_clip_out, num_classes=3)


def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img = ImageOps.grayscale(img)
        img_resized = np.array(img)
        # Apply DCT
        img_dct = dct(img_resized, type=2, norm="ortho", axis=0)
        img_dct = dct(img_dct, type=2, norm="ortho", axis=1)
        img_dct = np.abs(img_dct)
        img_dct += 1e-13
        img_dct = np.log(img_dct)
        img_dct -= np.mean(img_dct)
        img_dct /= np.std(img_dct)

        # Resize the DCT-transformed image to 128x128
        resized_dct = resize(img_dct, (128, 128, 1))
        images.append(resized_dct)

    return np.array(images)


# ### Test set
# Load real and fake images from folders
if args.mode=='COCO':
  real_images_test = load_images_from_folder('../data/COCO_test')
elif args.mode=='UCID':
  real_images_test = load_images_from_folder('../data/ucid')
elif args.mode=='ImageNet':
  real_images_test = load_images_from_folder('../data/imagenet')

real_images_man_test = load_images_from_folder('../data/test')
fake_images_df_test = load_images_from_folder('../data/deep_floyd')
fake_images_sd14_test = load_images_from_folder('../data/sd14')
fake_images_sd21_test = load_images_from_folder('../data/sd21')
fake_images_sdxl_test = load_images_from_folder('../data/sdxl')
fake_images_test = np.vstack((fake_images_sd14_test, fake_images_df_test, fake_images_sd21_test, fake_images_sdxl_test))

# Convert images to numpy arrays
X_real_test = np.array(real_images_test)
X_fake_test = np.array(fake_images_test)
X_man_test = np.array(real_images_man_test)

# Create labels for real and fake images
y_fake_test = np.ones(len(X_fake_test))
y_man_test = np.zeros(len(X_man_test))

# Combine real and fake images and labels
print(X_fake_test.shape, X_man_test.shape)
X_test = np.vstack((X_real_test, X_fake_test, X_man_test))

# Load man vs fake model
mvfmodel = load_model('../DCT_Classification/Models/manvsfake2207.h5')
mvfprob = mvfmodel.predict(X_test)
mvfout = np.round(mvfprob)

if args.mode=='COCO':
  a,b=200,1000,
elif args.mode=='UCID':
  a,b=886,1686
elif args.mode=='ImageNet':
  a,b=1000,1800

# Initialize variables for confusion matrix calculations
cr = cf = cm = rim = rif = fim = fir = mir = mif = 0
# Calculate confusion matrix components
for i in range(len(X_clip)):
    if i < a:  # True Real Images
        if three_class_clip_out_one_hot[i][2] == 1.0:  # Predicted as Manipulated
            cr += 1
        else:
            if mvfout[i][0] == 0:  # Predicted as Real
                fir += 1
            else:  # Predicted as Fake
                mir += 1
    elif i >= a and i < b:  # True Fake Images
        if three_class_clip_out_one_hot[i][2] == 1.0:  # Predicted as Manipulated
            rif += 1
        else:
            if mvfout[i][0] == 0:  # Predicted as Real
                cf += 1
            else:  # Predicted as Fake
                mif += 1
    elif i >= b:  # True Manipulated Images
        if three_class_clip_out_one_hot[i][2] == 1.0:  # Predicted as Manipulated
            rim += 1
        else:
            if mvfout[i][0] == 0:  # Predicted as Real
                fim += 1
            else:  # Predicted as Fake
                cm += 1

# Print confusion matrix
print("CONFUSION MATRIX")
print("     predicted real     predicted fake     predicted manipulated")
print(f"actual real           {cr}       {fir}       {mir}")
print(f"actual fake           {rif}       {cf}       {mif}")
print(f"actual manipulated    {rim}       {fim}       {cm}")

# Calculate precision, recall, and F1 score
def calculate_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

# Real class
TP_real = cr
FP_real = rif + rim
FN_real = fir + mir
precision_real, recall_real, f1_real = calculate_metrics(TP_real, FP_real, FN_real)

# Fake class
TP_fake = cf
FP_fake = fir + fim
FN_fake = rif + mif
precision_fake, recall_fake, f1_fake = calculate_metrics(TP_fake, FP_fake, FN_fake)

# Manipulated class
TP_man = cm
FP_man = mir + mif
FN_man = rim + fim
precision_man, recall_man, f1_man = calculate_metrics(TP_man, FP_man, FN_man)

# Print metrics
print("\nClassification Report:")
print(f"Class: Real - Precision: {precision_real:.2f}, Recall: {recall_real:.2f}, F1 Score: {f1_real:.2f}")
print(f"Class: Fake - Precision: {precision_fake:.2f}, Recall: {recall_fake:.2f}, F1 Score: {f1_fake:.2f}")
print(f"Class: Manipulated - Precision: {precision_man:.2f}, Recall: {recall_man:.2f}, F1 Score: {f1_man:.2f}")
