import cv2
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def show_image_with_caption(image_path, caption):
    """Display image with caption overlay"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, caption, (10, 30), font, 1, (255, 255, 255), 2)
    
    return Image.fromarray(img)
