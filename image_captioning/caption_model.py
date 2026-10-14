import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize BLIP model for image captioning
        """
        print("Loading BLIP model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()
    
    def generate_caption(self, image_path, max_length=50):
        """
        Generate caption for given image
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image and text
            inputs = self.processor(
                image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
            return caption
            
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def generate_multiple_captions(self, image_path, num_captions=3):
        """
        Generate multiple captions for same image
        """
        captions = []
        for i in range(num_captions):
            caption = self.generate_caption(image_path)
            captions.append(f"Caption {i+1}: {caption}")
        return captions