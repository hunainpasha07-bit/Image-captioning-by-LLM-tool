import streamlit as st
from PIL import Image
import os
from caption_model import ImageCaptioner

# Page config
st.set_page_config(
    page_title="🖼️ Image Captioning AI",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Captioning with LLM TOOL")
st.markdown("Upload an image and get AI-generated captions!")

# Initialize model
@st.cache_resource
def load_model():
    return ImageCaptioner()

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        # Generate caption
        if st.button("🎯 Generate Caption", type="primary"):
            with st.spinner("Generating caption..."):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                caption = model.generate_caption(temp_path)
                st.success("✅ Caption Generated!")
                st.markdown(f"**Caption:** {caption}")
                
                # Multiple captions
                if st.button("🔄 Generate 3 Different Captions"):
                    captions = model.generate_multiple_captions(temp_path)
                    for cap in captions:
                        st.markdown(f"• {cap}")
                
                # Clean up
                os.remove(temp_path)

# Sidebar with info
with st.sidebar:
    st.header("📊 Model Info")
    st.write("**Model:** LLM (Salesforce)")
    st.write("**Device:**", model.device)
    st.write("**Capabilities:**")
    st.write("• Zero-shot captioning")
    st.write("• High quality descriptions")
    st.write("• Works with any image")
    
    st.header("🚀 Quick Test")
    st.info("Upload any image above to test!")
