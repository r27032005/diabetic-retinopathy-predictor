"""
Prediction script for Diabetic Retinopathy Classification
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from config import Config

class DRPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model"""
        self.config = Config()
        self.model = load_model(model_path)
        print(f"Model loaded from: {model_path}")

    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.config.IMAGE_SIZE)
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)

            return image

        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image_path):
        """Predict diabetic retinopathy for a single image"""
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None, None, None

        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.config.CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]

        return predicted_class, confidence, predictions[0]

    def visualize_prediction(self, image_path, predicted_class, confidence, all_predictions):
        """Visualize prediction results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Display original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(image)
        ax1.set_title(f'Retinal Image\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}')
        ax1.axis('off')

        # Display prediction probabilities
        ax2.bar(self.config.CLASS_NAMES, all_predictions)
        ax2.set_title('Prediction Probabilities')
        ax2.set_ylabel('Probability')
        ax2.tick_params(axis='x', rotation=45)

        # Highlight predicted class
        max_idx = np.argmax(all_predictions)
        bars = ax2.patches
        bars[max_idx].set_color('red')

        plt.tight_layout()
        plt.savefig('prediction_result.png')
        print("Prediction visualization saved as 'prediction_result.png'")
        plt.show()

    def predict_batch(self, image_folder):
        """Predict for multiple images in a folder"""
        results = []

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                predicted_class, confidence, _ = self.predict(image_path)

                if predicted_class is not None:
                    results.append({
                        'filename': filename,
                        'prediction': predicted_class,
                        'confidence': confidence
                    })
                    print(f"{filename}: {predicted_class} ({confidence:.2%})")

        return results

def main():
    """Main prediction function"""
    import argparse

    parser = argparse.ArgumentParser(description='Predict Diabetic Retinopathy from retinal images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5 file)')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--folder', type=str, help='Path to folder containing multiple images')
    parser.add_argument('--visualize', action='store_true', help='Visualize prediction results')

    args = parser.parse_args()

    # Initialize predictor
    predictor = DRPredictor(args.model)

    if args.image:
        # Single image prediction
        print(f"\nPredicting for image: {args.image}")
        predicted_class, confidence, all_predictions = predictor.predict(args.image)

        if predicted_class is not None:
            print(f"\nPrediction: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print("\nAll class probabilities:")
            for i, class_name in enumerate(predictor.config.CLASS_NAMES):
                print(f"  {class_name}: {all_predictions[i]:.2%}")

            if args.visualize:
                predictor.visualize_prediction(args.image, predicted_class, confidence, all_predictions)

    elif args.folder:
        # Batch prediction
        print(f"\nPredicting for images in folder: {args.folder}")
        results = predictor.predict_batch(args.folder)
        print(f"\nProcessed {len(results)} images")

    else:
        print("Please provide either --image or --folder argument")

if __name__ == "__main__":
    # Example usage without command line arguments
    print("Diabetic Retinopathy Predictor")
    print("=" * 50)
    print("\nUsage examples:")
    print("1. Single image prediction:")
    print("   python predict.py --model models/saved_models/efficientnet_model_best.h5 --image path/to/image.jpg")
    print("\n2. Batch prediction:")
    print("   python predict.py --model models/saved_models/efficientnet_model_best.h5 --folder path/to/images/")
    print("\n3. With visualization:")
    print("   python predict.py --model models/saved_models/efficientnet_model_best.h5 --image path/to/image.jpg --visualize")
