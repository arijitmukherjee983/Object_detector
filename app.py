import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# Load model weights and categories
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img): 
    if img.mode != "RGB":
        img = img.convert("RGB")  # Ensure correct number of channels

    img_processed = img_preprocess(img)  # Transformed tensor: (3, H, W), float32
    prediction = model(img_processed.unsqueeze(0))[0]  # Dict with "boxes", "labels", "scores"
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img, dtype=torch.uint8)  # Ensure correct type
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]],
        width=2
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
    return img_with_bboxes_np

# Streamlit dashboard
st.title("Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)
    
    prediction = make_prediction(img)
    
    # Convert original image to (3, H, W) format for drawing boxes
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img).transpose(2, 0, 1)  # (H, W, 3) → (3, H, W)
    
    img_with_bbox = create_image_with_bboxes(img_array, prediction)

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img_with_bbox)
    plt.axis("off")
    st.pyplot(fig, use_container_width=True)

    del prediction["boxes"]  # Don't print raw boxes
    st.header("Predicted Classes")
    st.write(prediction)
