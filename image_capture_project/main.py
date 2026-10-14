import cv2
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class ImageCaptureApp:
    def __init__(self):
        # Create directory for captured images
        self.save_dir = "captured_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Initialize webcam (0 = default camera)
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera!")
            exit()
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.image_count = 0
        self.is_recording = False

    def capture_image(self):
        """Capture and save single image"""
        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/image_{self.image_count:03d}_{timestamp}.jpg"
            
            # Apply basic processing (optional)
            processed_frame = self.preprocess_image(frame)
            
            cv2.imwrite(filename, processed_frame)
            self.image_count += 1
            print(f"✅ Image saved: {filename}")
            return processed_frame
        return None

    def preprocess_image(self, frame):
        """Basic image preprocessing using OpenCV ML capabilities"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny (OpenCV ML algorithm)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to 3-channel for display
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Blend original and processed image
        blended = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        
        return blended

    def display_live_feed(self):
        """Display live camera feed with controls"""
        print("🎥 Image Capture App Started!")
        print("Controls:")
        print("  SPACEBAR - Capture Image")
        print("  'r' - Toggle Recording Mode")
        print("  'q' - Quit")
        print("  's' - Show captured images")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Add text overlay
            cv2.putText(frame, f"Captured: {self.image_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | R: Record | Q: Quit", (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Process frame for display
            display_frame = self.preprocess_image(frame)
            
            cv2.imshow('Image Capture - MLL Tool (OpenCV)', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar
                self.capture_image()
                
            elif key == ord('r') or key == ord('R'):  # Record toggle
                self.is_recording = not self.is_recording
                status = "ON" if self.is_recording else "OFF"
                print(f"📹 Recording mode: {status}")
                
            elif key == ord('s') or key == ord('S'):  # Show images
                self.show_captured_images()
                
            elif key == ord('q') or key == ord('Q'):  # Quit
                break
        
        self.cleanup()

    def show_captured_images(self):
        """Display all captured images in a gallery"""
        images = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        if not images:
            print("No images captured yet!")
            return
        
        print(f"📸 Showing {len(images)} captured images...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, img_file in enumerate(sorted(images)[-6:]):  # Show last 6
            if idx < len(axes):
                img_path = os.path.join(self.save_dir, img_file)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(img_file[:20], fontsize=10)
                axes[idx].axis('off')
        
        plt.suptitle('Captured Images Gallery', fontsize=16)
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        """Release camera and close windows"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Application closed successfully!")

def main():
    print("🚀 Starting Image Capture Project with OpenCV (MLL Tool)")
    print("=" * 50)
    
    app = ImageCaptureApp()
    app.display_live_feed()

if __name__ == "__main__":
    main()