import streamlit as st
from PIL import Image
import random
import requests


# Placeholder for your image captioning model function
def generate_caption(image, temperature):
    """
    Generate a caption for the given image using the specified temperature.

    Parameters:
        image: PIL Image object
        temperature: float (controls randomness in caption generation)

    Returns:
        str: Generated caption
    """
    api_url = "http://127.0.0.1:8000/caption"  # FastAPI URL
    try:
        # Save PIL image to bytes for API upload
        from io import BytesIO

        image_bytes = BytesIO()
        image.save(
            image_bytes, format="PNG"
        )  # Use appropriate format (e.g., JPEG, PNG)
        image_bytes.seek(0)

        # Make POST request to FastAPI
        files = {"image": ("uploaded_image.png", image_bytes, "image/png")}
        data = {"temperature": temperature}
        response = requests.post(api_url, files=files, data=data)

        # Parse and return the caption
        if response.status_code == 200:
            return response.json().get("caption", "No caption returned")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Failed to connect to API: {e}"


# Streamlit app
st.title("Image Captioning App")

st.sidebar.header("Settings")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Adjust the temperature to control caption randomness. Lower values produce deterministic captions.",
)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate and display caption
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, temperature)
        st.subheader("Generated Caption:")
        st.success(caption)

st.info(
    "Adjust the temperature slider in the sidebar to modify caption randomness before generating captions."
)
