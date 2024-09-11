import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.utils import to_categorical
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Synthetic Image Detection with 3-class classifier.')
parser.add_argument('--mode', type=str, default='COCO', help='Real image set.(COCO, UCID, ImageNet)')

# Parse arguments
args = parser.parse_args()

# Load precomputed CLIP encodings from the CSV file (use default if not provided)
if args.mode=='COCO':
  train_data = pd.read_csv('../data/CLIP_encodings_test/threewayenc_testcoco200.csv')
elif args.mode=='UCID':
  train_data = pd.read_csv('../data/CLIP_encodings_test/threewayenc_testucid.csv')
elif args.mode=='ImageNet':
  train_data = pd.read_csv('../data/CLIP_encodings_test/threewayenc_testimagenet.csv')


# Separate features and labels
X_test = data.iloc[:, 2:].values  # Assuming encodings start from the third column (index 2)
y_test = data.iloc[:, 1].values    # Assuming class is in the second column (index 1)

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded_test = label_encoder.fit_transform(y_test)

# Convert labels to one-hot encoding for AUC calculation
y_test_one_hot = to_categorical(y_encoded_test)


#  Define the model with Dropout layers
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dropout(0.5),  # Add Dropout to prevent overfitting
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),  # Add Dropout to prevent overfitting
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),  # Add Dropout to prevent overfitting
#     tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Three-class classification with softmax activation
#])

#  Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
#               metrics=['accuracy'])


model=tf.keras.models.load_model('../CLIP/3wayclip.h5')


# Get predictions
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

# Print classification report
print(classification_report(y_encoded_test, y_pred_classes, target_names=label_encoder.classes_))

# Calculate AUC for each class
auc = roc_auc_score(y_test_one_hot, y_pred_prob, multi_class="ovr")
print(f'AUC: {auc}')
