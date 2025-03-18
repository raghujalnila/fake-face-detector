from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = "binary_classification_model2.pth"
model = models.resnet18(pretrained=False)  # Load ResNet18
model.fc = nn.Linear(model.fc.in_features, 1)  # Ensure correct output layer
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()  # Convert logits to probability
    
    result = "Real Face" if prediction > 0.5 else "Fake Face"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
