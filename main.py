import streamlit as st
import os
from PIL import Image, UnidentifiedImageError
from modal_helper import predict  # Make sure this file contains the correct predict() function

# --- Helper Functions ---
def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to disk and return the file path."""
    file_path = "temp_file.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def validate_image(path: str) -> bool:
    """Check if the image at the given path is a valid image file."""
    try:
        img = Image.open(path)
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

# --- Main App ---
def main():
    st.set_page_config(page_title="Brain Tumour Classification", layout="centered")
    st.title("🧠 Brain Tumour Classification")

    st.markdown(
        """
        Upload an **MRI brain scan image**, and the model will classify it as either:
        - **Tumor (Yes)**
        - **No Tumor**

        ✅ Supported formats: **JPG** and **PNG**
        """
    )

    uploaded_file = st.file_uploader("📤 Upload an image file", type=["jpg", "png"])

    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file)

        if validate_image(image_path):
            st.success("✅ Image loaded and verified successfully!")

            # Display uploaded image (resized)
            img = Image.open(image_path)
            img_resized = img.resize((300, 300))
            st.image(img_resized, caption="Uploaded Image", use_column_width=False)

            # Predict using the loaded model
            with st.spinner("🔍 Classifying image..."):
                prediction = predict(image_path, device='cpu')

            st.info(f"🧠 **Predicted Class:** `{prediction}`")

            # Optional: clean up temp file
            os.remove(image_path)

        else:
            st.error("⚠️ Invalid image file. Please upload a valid JPG or PNG image.")

# --- Run App ---
if __name__ == "__main__":
    main()
