import cv2
import logging
import torch
from diffusers import StableDiffusionPipeline


# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class ImageSegmentation:
    """Handles image segmentation logic."""
    
    def process_image(self, image):
        """Dummy segmentation logic: Returns the input image."""
        return image

    def reset(self):
        """Clears both input and output images."""
        return None, None
    
    

class Imagesharpeness:
    """Handles image segmentation logic."""
    def __init__(self):
            pass
    
    def _load_model(self):
        model_path = r"models\ESPCN_x4.pb"
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        # self.sr.setModel(model_name="espcn", scale=4)
        self.sr.setModel("espcn", 4)
        logging.info(f"Loaded model: {model_path}")
        logging.info(f"Model is loaded: Success")
    def process_image(self, image):
        """Dummy segmentation logic: Returns the input image."""

        if image is None:
            raise ValueError("Error loading image. Check the file path.")
        
        self._load_model()

        # Apply super-resolution
        super_res_image = self.sr.upsample(image)
        
        return super_res_image


    def reset(self):
        """Clears both input and output images."""
        return None, None
    
    
class ImageGeneration:
    """Handles image generation logic."""
    def __init__(self):
        self._load_model()
        
    def reset(self):
        """Clears both input and output images."""
        return None, None
    
    def _load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device set to : ",self.device)
        self.model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(self.device)
        self.model.enable_attention_slicing()

    def generate_image(self,prompt: str, guidance_scale: float, width: int, height: int, inference_steps: int):
        # self._load_model()
        image = self.model(prompt, guidance_scale=guidance_scale, num_inference_steps=inference_steps).images[0]
        return image