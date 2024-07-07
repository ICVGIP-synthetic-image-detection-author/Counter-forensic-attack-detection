import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline
import numpy as np
import os
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")
# Load the pre-trained stable diffusion model from Hugging Face Model Hub
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
vae = pipe.vae
vae.eval()


if num_gpus > 1:
    vae = torch.nn.DataParallel(vae)
vae.to(device)

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize([0.5], [0.5])  # Normalize the image to match the input distribution of the model
])
# Load and preprocess the input image
# Replace the folder path with the appropriate path
for image_path in sorted(os.listdir('/kaggle/input/real-images-new/real'),key=extract_number):
    # input_image_path = '/kaggle/input/realimage/real_0.jpg'
    i=extract_number(image_path)
    input_image_path=(('/kaggle/input/real-images-new/real'+'/'+image_path))
    input_image = Image.open(input_image_path).convert("RGB")
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)
    # Clear cached memory
    torch.cuda.empty_cache()
    # Encode the image using the VAE
    with torch.no_grad():
        if num_gpus > 1:
            encoded = vae.module.encode(input_tensor)
        else:
            encoded = vae.encode(input_tensor)
        latents = encoded.latent_dist.sample()
    # Clear cached memory
    torch.cuda.empty_cache()
    # Decode the latents back to an image
    with torch.no_grad():
        if num_gpus > 1:
            decoded = vae.module.decode(latents)
        else:
            decoded = vae.decode(latents)
        reconstructed_image_tensor = decoded['sample']
    #print(type(reconstructed_image_tensor))
   # print((reconstructed_image_tensor))
    # Clear cached memory
    torch.cuda.empty_cache()
    #reconstructed_image_nparr=np.array(reconstructed_image_tensor)
    # Post-process the output tensor to a PIL image
    reconstructed_image_tensor = (reconstructed_image_tensor / 2 + 0.5).clamp(0, 1)  # Denormalize
    reconstructed_image_tensor = reconstructed_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_image = Image.fromarray((reconstructed_image_tensor * 255).astype(np.uint8))
    print("image reconstructed")
    print(type(reconstructed_image))
    # Save the reconstructed image
    reconstructed_image.save(f'reconstructed_image_{i}.jpg')