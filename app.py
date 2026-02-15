import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Team HyperBool | KrackHack 3.0", layout="wide")
st.title("🏜️ Autonomous Off-Road Semantic Segmentation")
st.markdown("Upload an image, and our model will download its pre-trained weights from the cloud to segment the terrain in real-time.")

# --- DYNAMIC COLOR MAPPING ---
# Nvidia's cloud model outputs 150 classes. We generate a distinct color palette for all 150.
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(150, 3), dtype=np.uint8)

# --- INFERENCE PIPELINE ---
@st.cache_resource
def load_model():
    """Downloads and caches the pre-trained weights directly from HuggingFace."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # THE MAGIC FIX: This fetches the model and processor directly from the internet.
    # No local .pth files required!
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    
    model.to(device)
    model.eval()
    return processor, model, device

def decode_mask(mask):
    """Converts the 2D class index array into an RGB image."""
    return COLORS[mask]

# --- APP LOGIC ---
processor, model, device = load_model()

uploaded_file = st.file_uploader("Upload an Off-Road Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Load image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    # 2. Preprocess using HuggingFace's built-in cloud processor
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    
    # 3. Predict
    with st.spinner("Fetching cloud weights and analyzing terrain..."):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Upsample back to the original image's size for a perfect overlay
            logits = F.interpolate(logits, size=image_pil.size[::-1], mode='bilinear', align_corners=False)
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
    # 4. Colorize and blend
    color_mask = decode_mask(pred_mask)
    blended = cv2.addWeighted(image_np, 0.6, color_mask, 0.4, 0)
    
    # 5. Extract unique classes found in the image
    unique_classes = np.unique(pred_mask)
    detected_labels = [model.config.id2label[c] for c in unique_classes]
    
    # 6. UI Display
    st.success(f"Segmentation Complete! Detected elements: **{', '.join(detected_labels)}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Input RGB")
        st.image(image_pil, use_container_width=True)
    with col2:
        st.header("Semantic Mask")
        st.image(color_mask, use_container_width=True)
    with col3:
        st.header("Blended Overlay")
        st.image(blended, use_container_width=True)