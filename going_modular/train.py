"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import argparse
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Create a parser
parser = argparse.ArgumentParser(description="Get hyperparameters")

# Get arg for hyperparameters
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train the model")
parser.add_argument("--batch_size", default=32, type=int, help="number of samples per batch")
parser.add_argument("--hidden_units", default=10, type=int, help="number of hidden units in hidden layers")
parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate to train the model")
parser.add_argument("--train_dir", default="data/pizza_steak_sushi/train", type=str, help="directory file path to training data")
parser.add_argument("--test_dir", default="data/pizza_steak_sushi/test", type=str, help="directory file path to testing data")

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file path: {train_dir}")
print(f"[INFO] Testing data file path: {test_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# Create dataloaders using data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create a model using model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

# Start training using engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=EPOCHS,
             device=device)

# Save the model using utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
