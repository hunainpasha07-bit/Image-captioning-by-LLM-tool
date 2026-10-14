import gradio as gr
from caption_model import ImageCaptioner

model = None

def get_model():
    global model
    if model is None:
        model = ImageCaptioner()
    return model


def caption_image(image):
    if image is None:
        return "Please upload an image!"
    
    # Save image temporarily
    image.save("temp_image.jpg")
    caption = get_model().generate_caption("temp_image.jpg")
    return caption

# Create Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Image Captioning Demo",
    description="Upload an image to get AI-generated captions using LLM model!"
)

if __name__ == "__main__":
    iface.launch(share=True)