from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
import logging
import json
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CLIPService")

# FastAPI app
app = FastAPI()

# Load CLIP model and processor
with open("config.json", "r") as f:
    config = json.load(f)
MODEL_PATH = config["model_paths"]["clip"]["base"]

model = CLIPModel.from_pretrained(MODEL_PATH)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class ImageRequest(BaseModel):
    image: str  # base64 encoded image

@app.post("/embed")
async def embed_image(request: ImageRequest):
    """Get CLIP embedding for an image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image.split(",")[-1])
        image = Image.open(BytesIO(image_data))
        
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get image features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Convert to numpy and normalize
        embedding = image_features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return {
            "embedding": embedding.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()
