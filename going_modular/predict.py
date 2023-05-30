
import torch
import torchvision
import argparse
import model_builder

# Creating a parser
parser = argparse.ArgumentParser()

# Get an image path
parser.add_argument("--image", help="target image to predict on")

# Get a model path
parser.add_argument("--model_path", default="models/05_going_modular_tingvgg_model.pth", help="target model to use for prediction")

args = parser.parse_args()

# Setup class names
class_names = ["pizza", "steak", "sushi"]

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the image path
IMG_PATH = args.image
print(f"[INFO] Predicting on {IMG_PATH}")

# Function to load in the model
def load_model(filepath=args.model_path):
    # Need to use same hyperparameters as the saved model
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=10,
                                  output_shape=3).to(device)
    print(f"[INFO] Loading in model from: {filepath}")                              
    # Load in the saved model state dictionary from file                                  
    model.load_state_dict(torch.load(filepath))

    return model

# Function to load in model + predict on select image
def predict_on_image(image=IMG_PATH, filepath=args.model_path):
    # Load the model
    model = load_model(filepath)
    # Load in the image and turn it into torch.float32 (same type as model)
    image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

    # Preprocess the image to get it between 0 and 1
    image = image / 255.

    # Expand to 4 dims to include batch dimension [batch_size, color_channels, height, width]
    image = image.unsqueeze(0)

    # Resize the image to be on the same size as the model
    transform = torchvision.transforms.Resize(size=(64, 64), antialias=True)
    image = transform(image)

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put image to target device
        image = image.to(device)

        # Get pred logits
        pred_logits = model(image)

        # Get pred probs
        pred_probs = torch.softmax(pred_logits, dim=1)

        # Get pred labels
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_label_class = class_names[pred_label]

    print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_probs.max():.3f}")

# Indicating that when you pass the file, it'll be the main one
if __name__=="__main__":
    predict_on_image()
