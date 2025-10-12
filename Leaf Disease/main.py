import os
import json
import logging
import sys
import base64
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from transformers import pipeline
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DiseaseAnalysisResult:
    disease_detected: bool
    disease_name: Optional[str]
    disease_type: str
    severity: str
    confidence: float
    symptoms: List[str]
    possible_causes: List[str]
    treatment: List[str]
    analysis_timestamp: str = datetime.now().astimezone().isoformat()


class LeafDiseaseDetector:
    """
    Leaf Disease Detector using open-source Hugging Face Vision models.
    """

    def __init__(self):
        load_dotenv()
        logger.info("Initializing free Vision Transformer model...")
        # Load a general-purpose vision model
        self.model = pipeline("image-classification", model="microsoft/resnet-50")
        logger.info("Model loaded successfully.")

    def analyze_leaf_image_base64(self, base64_image: str) -> Dict:
        try:
            logger.info("Starting analysis...")

            if not isinstance(base64_image, str):
                raise ValueError("base64_image must be a string")

            if base64_image.startswith('data:'):
                base64_image = base64_image.split(',', 1)[1]

            # Decode base64 string to image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))

            # Run prediction
            predictions = self.model(image)
            top_pred = predictions[0]  # top result
            label = top_pred['label'].lower()
            confidence = top_pred['score'] * 100

            # Simple disease detection logic (customize this part)
            if "leaf" not in label and "plant" not in label:
                result = DiseaseAnalysisResult(
                    disease_detected=False,
                    disease_name=None,
                    disease_type="invalid_image",
                    severity="none",
                    confidence=confidence,
                    symptoms=["This image does not contain a plant leaf"],
                    possible_causes=["Invalid image type uploaded"],
                    treatment=["Please upload an image of a plant leaf"]
                )
            else:
                result = DiseaseAnalysisResult(
                    disease_detected=True,
                    disease_name="Generic Leaf Condition",
                    disease_type="fungal" if "spot" in label else "unknown",
                    severity="moderate",
                    confidence=confidence,
                    symptoms=["Discoloration", "Possible fungal infection"],
                    possible_causes=["Humidity", "Nutrient imbalance"],
                    treatment=["Use fungicide", "Ensure proper drainage"]
                )

            return result.__dict__

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise


def main():
    try:
        detector = LeafDiseaseDetector()
        print("✅ Leaf Disease Detector initialized successfully.")
        print("Use analyze_leaf_image_base64() with a base64 image string.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
