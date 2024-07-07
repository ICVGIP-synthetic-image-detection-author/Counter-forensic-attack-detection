import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv('/home/deepfakedetect/image_encodings_train.csv')

#Filter the data to include only samples from class 0 and class 1
#filtered_data = data[data['class'].isin(['real','df', 'sd21','sd14'])]


X = data.drop(columns=['class']).values
y = (data['class'] == 'real').astype(int).values  # Class 0 -> 0, Class 1 -> 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(512,)),
    tf.keras.layers.Dropout(0.5),  # Add Dropout to prevent overfitting
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add Dropout to prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add Dropout to prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification with sigmoid activation
])

#Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
              metrics=['accuracy'])

# Step 7: Define EarlyStopping callback to monitor validation accuracy and prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Step 8: Train the model with EarlyStopping callback
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Step 9: Evaluate the model on test data and print test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Step 10: Evaluate the model and print classification report
y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred).flatten().astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))
#model.save('/home/deepfakedetect/deepfakedetect/classifiers/mlp_clip_classifier.h5')