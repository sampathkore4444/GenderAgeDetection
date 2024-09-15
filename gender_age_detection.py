import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define human-related labels and general labels for fallback
human_labels = [
    "a newborn baby girl",
    "a newborn baby boy",
    "an infant girl",
    "an infant boy",
    "a young girl",
    "a young boy",
    "an adult woman",
    "an adult man",
    "an elderly woman",
    "an elderly man"
]

fallback_labels = [
    "a human",
    "an animal",
    "an object"
]

# Function to classify image with fallback
def classify_image(image):
    # Prepare the image and labels for the model
    inputs = processor(images=image, text=fallback_labels, return_tensors="pt", padding=True)
    
    # Perform classification for general content
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate similarities and find the best match
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    top_probs, top_indices = probs.topk(1, dim=1)
    fallback_label = fallback_labels[top_indices[0][0].item()]
    confidence = top_probs[0][0].item()
    
    if fallback_label == "a human":
        # Proceed with specific human classification if a human is detected
        inputs = processor(images=image, text=human_labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        top_probs, top_indices = probs.topk(1, dim=1)
        label = human_labels[top_indices[0][0].item()]
        confidence = top_probs[0][0].item()
        return label, confidence
    else:
        return fallback_label, confidence

# Streamlit UI
st.title("Gender and Age Analyzer App powered by AI")
st.write("Upload an image to classify the gender and age group, or get general information.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Display loading message
        with st.spinner("Classifying the image..."):
            label, confidence = classify_image(image)
        
        st.write(f"**Classification Result:** {label}")
        st.write(f"**Confidence Score:** {confidence:.2f}")
        
        # Provide option to download the classified image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(label="Download Classified Image", data=buf.getvalue(), file_name="classified_image.png")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Optional: Provide a feedback form for users
st.subheader("Provide Feedback")
feedback = st.text_area("Please provide any feedback or report incorrect classifications:")
if st.button("Submit Feedback"):
    st.write("Thank you for your feedback!")
