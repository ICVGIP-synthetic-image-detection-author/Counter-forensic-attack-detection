
import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.transform import resize
from transformers import CLIPProcessor, CLIPModel
import torch

def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])


# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Function to extract image encodings and class values from a folder
def extract_image_encodings_from_folder(folder_path, class_value):
    failed=[]
    encodings=[]
    for filename in sorted(os.listdir(folder_path),key=extract_number):
        
        i=extract_number(filename)
        
        print(filename)
        
        try:
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                # image_resized=resize(np.array(image),(128,128), anti_aliasing=True)
                # resized_shape=np.shape((np.array(image)))
                # if resized_shape!=(128,128,3):
                #     failed.append(filename)
                #     continue
                inputs = processor(images=image    , return_tensors="pt", padding=True)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    # print("image feature shape",image_features.shape())
                # Remove unnecessary dimensions
                image_encoding = image_features.squeeze().tolist()
                
                encodings.append([class_value] + image_encoding)
        except:
            failed.append(filename)

    return encodings

# Define folder paths and class values
folder_paths = {
    'real': '/home/deepfakedetect/deepfakedetect/train2/real',
    'df': '/home/deepfakedetect/deepfakedetect/train2/deep_floyd',
    'Sd14': '/home/deepfakedetect/deepfakedetect/train2/sd14',
    'sd21': '/home/deepfakedetect/deepfakedetect/train2/sd21',
    'sdxl': '/home/deepfakedetect/deepfakedetect/train2/sdxl'
}

# Initialize list to store encodings and class values
data = []

# Loop through each folder and extract encodings
for class_value, folder_path in folder_paths.items():
    encodings = extract_image_encodings_from_folder(folder_path, class_value)
    data.extend(encodings)

# Convert data to DataFrame
df = pd.DataFrame(data, columns=['class'] + [f'encoding_{i}' for i in range(512)])


# Save DataFrame to CSV file
df.to_csv('image_encodings_train.csv', index=False)     


##FOR TEST CSV
# Define folder paths and class values
folder_paths_test= {
    'real': '/home/deepfakedetect/deepfakedetect/test/real',
    'df': '/home/deepfakedetect/deepfakedetect/test/deep_floyd',
    'Sd14': '/home/deepfakedetect/deepfakedetect/test/sd14',
    'sd21': '/home/deepfakedetect/deepfakedetect/test/sd21',
    'sdxl': '/home/deepfakedetect/deepfakedetect/test/sdxl'
}

# Initialize list to store encodings and class values
test_data = []

# Loop through each folder and extract encodings
for class_value, folder_path in folder_paths_test.items():
    encodings = extract_image_encodings_from_folder(folder_path, class_value)
    test_data.extend(encodings)

# Convert data to DataFrame
test_df = pd.DataFrame(test_data, columns=['class'] + [f'encoding_{i}' for i in range(512)])


# Save DataFrame to CSV file
test_df.to_csv('image_encodings_test.csv', index=False)   
