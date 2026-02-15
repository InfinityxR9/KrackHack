import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PRETRAINED_MODEL = "nvidia/segformer-b1-finetuned-ade-512-512"
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Color Palette (Mapped exactly to KrackHack PDF)
# --------------------------------------------------

CLASS_COLORS = {
    0: (34, 139, 34),     # Trees (Forest Green)
    1: (0, 255, 0),       # Lush Bushes (Lime)
    2: (189, 183, 107),   # Dry Grass (Dark Khaki)
    3: (139, 69, 19),     # Dry Bushes (Saddle Brown)
    4: (128, 128, 128),   # Ground Clutter (Gray)
    5: (255, 20, 147),    # Flowers (Deep Pink)
    6: (160, 82, 45),     # Logs (Sienna)
    7: (220, 20, 60),     # Rocks (Crimson)
    8: (244, 164, 96),    # Landscape/Sand (Sandy Brown)
    9: (135, 206, 235),   # Sky (Sky Blue)
}

CLASS_NAMES = {
    0: "Trees",
    1: "Lush Bushes",
    2: "Dry Grass",
    3: "Dry Bushes",
    4: "Ground Clutter",
    5: "Flowers",
    6: "Logs",
    7: "Rocks",
    8: "Landscape (Sand)",
    9: "Sky",
}

# --------------------------------------------------
# Model Wrapper (Required)
# --------------------------------------------------

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

# --------------------------------------------------
# Cached Model Loader
# --------------------------------------------------

@st.cache_resource
def load_model():
    processor = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL)
    model = SegFormerWrapper(PRETRAINED_MODEL, NUM_CLASSES)
    model.to(DEVICE)
    model.eval()
    return processor, model

# --------------------------------------------------
# Decode Mask to RGB
# --------------------------------------------------

def decode_segmentation(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(layout="wide")
st.title("🏜️ Off-Road Desert Semantic Segmentation")
st.write("Upload a desert image to perform semantic segmentation. Classes are mapped strictly to Team HyperBool's custom dataset.")

uploaded_file = st.file_uploader("Upload JPG/PNG Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    processor, model = load_model()

    # --------------------------
    # Load & Resize Image
    # --------------------------
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((512, 512)) # Model Input Size
    image_np = np.array(image)

    # --------------------------
    # Preprocess
    # --------------------------
    inputs = processor(
        images=image,
        return_tensors="pt",
        do_resize=False
    )
    pixel_values = inputs["pixel_values"].to(DEVICE)

    # --------------------------
    # Inference
    # --------------------------
    with torch.no_grad():
        outputs = model(pixel_values)
        predictions = torch.argmax(outputs, dim=1)
        predicted_mask = predictions.squeeze().cpu().numpy()

    # --------------------------
    # Colorize & Overlay
    # --------------------------
    color_mask = decode_segmentation(predicted_mask)
    overlay = cv2.addWeighted(image_np, 0.6, color_mask, 0.4, 0)

    # --------------------------
    # Calculate Percentages
    # --------------------------
    total_pixels = predicted_mask.size
    class_percentages = {}
    
    for class_id, name in CLASS_NAMES.items():
        pixel_count = np.sum(predicted_mask == class_id)
        if pixel_count > 0:
            pct = (pixel_count / total_pixels) * 100
            class_percentages[name] = pct
            
    # Sort from highest percentage to lowest
    sorted_percentages = sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)

    # --------------------------
    # Display Visual Results
    # --------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original (512x512)")
        st.image(image_np, use_column_width=True)

    with col2:
        st.subheader("Semantic Mask")
        st.image(color_mask, use_column_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_column_width=True)

    st.divider()

    # --------------------------
    # Display Percentages & Legend
    # --------------------------
    col_metrics, col_legend = st.columns([1, 1])

    with col_metrics:
        st.subheader("📊 Terrain Composition (%)")
        for name, pct in sorted_percentages:
            # Display text and a visual progress bar
            st.write(f"**{name}:** {pct:.1f}%")
            st.progress(int(pct) if int(pct) > 0 else 1)

    with col_legend:
        st.subheader("🗺️ Class Legend")
        legend_cols = st.columns(3)
        for idx, (class_id, name) in enumerate(CLASS_NAMES.items()):
            color = CLASS_COLORS[class_id]
            # Convert RGB tuple to Hex for HTML styling
            hex_color = '#%02x%02x%02x' % color
            with legend_cols[idx % 3]:
                st.markdown(
                    f"<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
                    f"<div style='width: 20px; height: 20px; background-color: {hex_color}; margin-right: 10px; border-radius: 3px;'></div>"
                    f"<span>{name}</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )