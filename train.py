"""
Training script for Diabetic Retinopathy Classification
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from data_preprocessing import DataPreprocessor
from model_architecture import DRModel
from config import Config

class DRTrainer:
    def __init__(self):
        self.config = Config()
        self.preprocessor = DataPreprocessor()
        self.dr_model = DRModel()

    def prepare_data(self, csv_file, images_dir):
        """Prepare training, validation, and test datasets"""
        print("Preparing datasets...")

        train_df, val_df, test_df = self.preprocessor.prepare_dataset(csv_file, images_dir)

        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")

        print("\nClass distribution in training set:")
        print(train_df['diagnosis'].value_counts())

        return train_df, val_df, test_df

    def train_model(self, train_df, val_df, model_type='efficientnet'):
        """Train the model"""
        print(f"\nTraining {model_type.upper()} Model...")

        # Create data generators
        train_gen, val_gen = self.preprocessor.create_data_generators(train_df, val_df)

        # Create and compile model
        if model_type == 'custom':
            model = self.dr_model.create_custom_cnn()
        else:
            model, base_model = self.dr_model.create_efficientnet_model()

        model = self.dr_model.compile_model(model)

        print("Model Architecture:")
        model.summary()

        # Get callbacks
        callbacks = self.dr_model.get_callbacks(f'{model_type}_model')

        # Train model
        history = model.fit(
            train_gen,
            epochs=self.config.EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        return model, history

    def evaluate_model(self, model, test_df):
        """Evaluate the trained model"""
        print("\nEvaluating model...")

        test_gen = self.preprocessor.create_data_generators(test_df)

        predictions = model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)

        true_labels = test_gen.classes

        print("\nClassification Report:")
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=self.config.CLASS_NAMES,
            digits=4
        )
        print(report)

        cm = confusion_matrix(true_labels, predicted_classes)
        self.plot_confusion_matrix(cm, self.config.CLASS_NAMES)

        return predictions, predicted_classes, true_labels

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")

    def run_training(self, csv_file, images_dir, model_type='efficientnet'):
        """Run the complete training pipeline"""
        print("Starting Diabetic Retinopathy Model Training...")
        print("=" * 60)

        # Create model directory
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

        # Prepare data
        train_df, val_df, test_df = self.prepare_data(csv_file, images_dir)

        # Train model
        model, history = self.train_model(train_df, val_df, model_type)

        # Evaluate model
        predictions, predicted_classes, true_labels = self.evaluate_model(model, test_df)

        print("\nTraining completed successfully!")
        return model, history

def main():
    """Main training function"""
    trainer = DRTrainer()

    csv_file = 'data/labels/train.csv'
    images_dir = 'data/train_images/'

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        print("Please download the APTOS 2019 dataset from Kaggle.")
        return

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        print("Please download the APTOS 2019 dataset from Kaggle.")
        return

    try:
        model, history = trainer.run_training(
            csv_file=csv_file,
            images_dir=images_dir,
            model_type='efficientnet'
        )

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
