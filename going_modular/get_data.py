
# Import libraries
import os
import requests
import zipfile
import torch
import torchvision
from torchvision import datasets
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
zipfile_path = data_path / "pizza_steak_sushi.zip"

# If the image folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} already exists")
else:
    print(f"Did not find {image_path} directory, creating one")
    image_path.mkdir(parents=True, exist_ok=True)

# Download data
with open(zipfile_path, "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading data")
    f.write(request.content)

# Unzip data
with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
    print(f"Unzipping data")
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(zipfile_path)

# To get the original dataset from torchvision.datasets
# food101_data_train = torchvision.datasets.Food101(root=image_path, split='train', download=True)
# food101_data_test = datasets.Food101(root=image_path, split='test', download=True)
