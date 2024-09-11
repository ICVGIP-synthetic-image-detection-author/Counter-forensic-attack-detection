import numpy as np
import os
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

parser = argparse.ArgumentParser(description='Synthetic Image Detection with 3-class classifier.')
parser.add_argument('--real', type=str, default='COCO', help='Real image set. (COCO (default), UCID or ImageNet)')
parser.add_argument('--mode', type=str, default='rf', help='real vs. fake, real vs. manipulated or fake vs. manipulated. (rf (default), rm or fm)')


def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img=ImageOps.grayscale(img)
        img_resized=np.array(img)
        # Apply DCT
        img_dct = dct(img_resized, type=2, norm="ortho", axis=0)
        img_dct = dct(img_dct, type=2, norm="ortho", axis=1)
        img_dct = np.abs(img_dct)
        img_dct += 1e-13
        img_dct = np.log(img_dct)
        img_dct -= np.mean(img_dct)
        img_dct /= np.std(img_dct)

        # # Resize the DCT-transformed image to 128x128
        resized_dct=resize(img_dct,(128,128,1))

        images.append(resized_dct)

    return np.array(images)


###TEST DATA
# Load images from three folders representing three classes
if args.mode=='rf':
  if args.real=='COCO':
    real_images_test = load_images_from_folder('../data/COCO_test')
  elif args.real=='UCID':
    real_images_test = load_images_from_folder('../data/ucid')
  elif args.real=='ImageNet':
    real_images_test = load_images_from_folder('../data/imagenet')
  fake_images_df_test = load_images_from_folder('../data/deep_floyd')
  fake_images_sd14_test = load_images_from_folder('../data/sd14')
  fake_images_sd21_test = load_images_from_folder('../data/sd21')
  fake_images_sdxl_test = load_images_from_folder('../data/sdxl')

elif args.mode=='rm':
  if args.real=='COCO':
    real_images_test = load_images_from_folder('../data/COCO_test')
  elif args.real=='UCID':
    real_images_test = load_images_from_folder('../data/ucid')
  elif args.real=='ImageNet':
    real_images_test = load_images_from_folder('../data/imagenet')
  
  real_images_man_test = load_images_from_folder('../data/test')

elif args.mode=='fm':
  fake_images_df_test = load_images_from_folder('../data/deep_floyd')
  fake_images_sd14_test = load_images_from_folder('../data/sd14')
  fake_images_sd21_test = load_images_from_folder('../data/sd21')
  fake_images_sdxl_test = load_images_from_folder('../data/sdxl')
 
  real_images_man_test = load_images_from_folder('../data/test')



# Convert images to numpy arrays
if args.mode=='rf':
  if args.real=='COCO':
    X_real_test=np.reshape(real_images_test,(200,128,128,1))
  elif args.real=='UCID':
    X_real_test=np.reshape(real_images_test,(886,128,128,1))
  elif args.real=='ImageNet':
    X_real_test=np.reshape(real_images_test,(1000,128,128,1))

  print(X_real_test.shape)
  X_fake_test = np.vstack((fake_images_df_test,fake_images_sd14_test,fake_images_sd21_test,fake_images_sdxl_test))
  print(X_fake_test.shape)

elif args.mode=='rm':
  if args.real=='COCO':
    X_real_test=np.reshape(real_images_test,(200,128,128,1))
  elif args.real=='UCID':
    X_real_test=np.reshape(real_images_test,(886,128,128,1))
  elif args.real=='ImageNet':
    X_real_test=np.reshape(real_images_test,(1000,128,128,1))
  X_man = np.vstack(real_images_man_test)
  X_man_test=np.reshape(X_man,(200,128,128,1))
  print(X_man_test.shape)

elif args.mode=='fm':
  X_fake_test = np.vstack((fake_images_df_test,fake_images_sd14_test,fake_images_sd21_test,fake_images_sdxl_test))
  print(X_fake_test.shape)
  X_man = np.vstack(real_images_man_test)
  X_man_test=np.reshape(X_man,(200,128,128,1))
  print(X_man_test.shape)

# Create labels for each class
if args.mode=='rf':
  y_real_test = np.ones(len(X_real_test))
  y_fake_test = np.zeros(len(X_fake_test))
  # Combine images and labels from all classes
  X_test = np.vstack((X_real_test, X_fake_test))
  y_test = np.hstack((y_real_test, y_fake_test))
elif args.mode=='rm':
  y_real_test = np.ones(len(X_real_test))
  y_man_test = np.zeros(len(X_man_test))
  X_test = np.vstack((X_real_test, X_man_test))
  y_test = np.hstack((y_real_test, y_man_test)) 
elif args.mode=='fm':
  y_man_test = np.ones(len(X_man_test))
  y_fake_test = np.zeros(len(X_fake_test)) 
  X_test = np.vstack((X_man_test, X_fake_test))
  y_test = np.hstack((y_man_test, y_fake_test))




early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


# history = model.fit(X_train, y_train, epochs=20, batch_size=16)
from keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

if args.mode=='rf':
  model=load_model('../DCT_Classification/Models/ICVGIP_dctrvf.h5')
elif args.mode=='rm':
  model=load_model('../DCT_Classification/Models/manvsreal2207cocofixedman.h5')
elif args.mode=='fm':
  model=load_model('../DCT_Classification/Models/manvsfake2207.h5')

# Predict class probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class predictions
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate AUC for each class
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print("AUC:", auc)

if args.mode=='rf':
  class_names = ["real", "fake"]
elif args.mode=='rm':
  class_names = ["real", "manipulated"]
elif args.mode=='fm':
  class_names = ["fake", "manipulated"]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
