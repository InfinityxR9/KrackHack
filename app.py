import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# -------------------------------
# Configuration
# -------------------------------

PRETRAINED_MODEL = "nvidia/segformer-b1-finetuned-ade-512-512"
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Color Palette (10 classes)
# -------------------------------

CLASS_COLORS = {
    0: (0, 0, 0),          # Background
    1: (128, 64, 128),     # Road
    2: (244, 35, 232),     # Sand
    3: (70, 70, 70),       # Rock
    4: (102, 102, 156),    # Vegetation
    5: (190, 153, 153),    # Sky
    6: (153, 153, 153),    # Vehicle
    7: (250, 170, 30),     # Human
    8: (220, 220, 0),      # Obstacle
    9: (107, 142, 35),     # Terrain
}

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Sand",
    3: "Rock",
    4: "Vegetation",
    5: "Sky",
    6: "Vehicle",
    7: "Human",
    8: "Obstacle",
    9: "Terrain",
}

# -------------------------------
# Model Wrapper (REQUIRED)
# -------------------------------

class SegFormerWrapper(nn.Module):
    def __init__(self, pretrained_name, num_classes):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        return logits

# -------------------------------
# Load Model (Cached)
# -------------------------------

@st.cache_resource
def load_model():
    processor = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model = SegFormerWrapper(PRETRAINED_MODEL, NUM_CLASSES)
    model.to(DEVICE)
    model.eval()
    return processor, model

# -------------------------------
# Mask Coloring Function
# -------------------------------

def decode_segmentation(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(layout="wide")
st.title("🏜️ Off-Road Desert Semantic Segmentation (SegFormer-B1)")
st.write("Upload a desert image to perform semantic segmentation.")

uploaded_file = st.file_uploader("Upload JPG/PNG Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    processor, model = load_model()

    # Load Image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(pixel_values)
        predictions = torch.argmax(outputs, dim=1)
        predicted_mask = predictions.squeeze().cpu().numpy()

    # Convert mask to color
    color_mask = decode_segmentation(predicted_mask)

    # Create overlay using OpenCV
    overlay = cv2.addWeighted(
        image_np,
        0.6,
        color_mask,
        0.4,
        0
    )

    # Display Results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True)

    with col2:
        st.subheader("Semantic Mask")
        st.image(color_mask, use_column_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_column_width=True)

    # -------------------------------
    # Legend
    # -------------------------------

    st.subheader("🗺️ Class Legend")

    legend_cols = st.columns(5)
    for idx, (class_id, name) in enumerate(CLASS_NAMES.items()):
        color = CLASS_COLORS[class_id]
        color_box = np.ones((40, 40, 3), dtype=np.uint8)
        color_box[:] = color

        with legend_cols[idx % 5]:
            st.image(color_box, width=40)
            st.write(name)