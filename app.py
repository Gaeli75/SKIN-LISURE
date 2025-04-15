import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Make sure this is the first Streamlit command
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")

# Load class labels (from your training data)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Update if needed

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Preprocessing (same as your eval_transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# UI
st.title("Skin Lesion Classifier")
st.write("Upload or capture an image of a skin lesion for prediction.")

input_method = st.radio("Choose input method:", ["Upload", "Take a picture"])
img_file = None

if input_method == "Upload":
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
else:
    img_file = st.camera_input("Take a picture")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        label = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100

    # Display prediction
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence_percent:.2f}%")

    # Show warning if confidence > 75%
    if confidence_percent > 75:
        st.warning("⚠️ High confidence! It's strongly recommended to consult a dermatologist.")

