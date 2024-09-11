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
parser.add_argument('--mode', type=str, default='COCO', help='Real image set. (COCO, UCID or ImageNet)')


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
if args.mode=='COCO':
  real_images_test = load_images_from_folder('../data/COCO_test')
elif args.mode=='UCID':
  real_images_test = load_images_from_folder('../data/ucid')
elif args.mode=='ImageNet':
  real_images_test = load_images_from_folder('../data/imagenet')
print(len(real_images_test))
real_images_man_test = load_images_from_folder('../data/test')
fake_images_df_test = load_images_from_folder('../data/deep_floyd')
fake_images_sd14_test = load_images_from_folder('../data/sd14')
fake_images_sd21_test = load_images_from_folder('../data/sd21')
fake_images_sdxl_test = load_images_from_folder('../data/sdxl')

# Convert images to numpy arrays
if args.mode=='COCO':
  X_real_test=np.reshape(real_images_test,(200,128,128,1))
elif args.mode=='UCID':
  X_real_test=np.reshape(real_images_test,(886,128,128,1))
elif args.mode=='ImageNet':
  X_real_test=np.reshape(real_images_test,(1000,128,128,1))

print(X_real_test.shape)
X_fake_test = np.vstack((fake_images_df_test,fake_images_sd14_test,fake_images_sd21_test,fake_images_sdxl_test))
print(X_fake_test.shape)
X_man = np.vstack(real_images_man_test)
X_man_test=np.reshape(X_man,(200,128,128,1))
print(X_man_test.shape)

# Create labels for each class
y_real_test = np.zeros(len(X_real_test))
y_fake_test = np.ones(len(X_fake_test))
y_man_test = np.full(len(X_man_test), 2)

# Combine images and labels from all classes
X_test = np.vstack((X_real_test, X_fake_test, X_man_test))
y_test = np.hstack((y_real_test, y_fake_test, y_man_test))


early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


# history = model.fit(X_train, y_train, epochs=20, batch_size=16)
from keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Load the model
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),  
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # Adjust output layer if needed
])

# Load the weights into the model
model.load_weights('/content/drive/MyDrive/dct_manipulated_detectors/fixed_artifact_detector_3typemulticlass_coco.h5')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Predict class probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class predictions
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate AUC for each class
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print("AUC:", auc)

class_names = ["real", "fake", "manipulated"]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
