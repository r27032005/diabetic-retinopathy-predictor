"""
Data preprocessing module for diabetic retinopathy images
"""

import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.config = Config()

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess a single retinal image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.crop_image_from_gray(image)
            image = cv2.resize(image, target_size)
            image = self.apply_clahe(image)
            image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def crop_image_from_gray(self, img, tol=7):
        """Crop out black borders from retinal images"""
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

    def apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)
        image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return image_clahe

    def create_data_generators(self, train_df, validation_df=None):
        """Create data generators for training and validation"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='diagnosis',
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )

        if validation_df is not None:
            val_generator = val_datagen.flow_from_dataframe(
                validation_df,
                x_col='image_path',
                y_col='diagnosis',
                target_size=self.config.IMAGE_SIZE,
                batch_size=self.config.BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
            return train_generator, val_generator

        return train_generator

    def prepare_dataset(self, csv_path, images_dir):
        """Prepare the dataset from CSV file containing image paths and labels"""
        df = pd.read_csv(csv_path)
        df['image_path'] = df['id_code'].apply(lambda x: os.path.join(images_dir, f"{x}.png"))
        df['diagnosis'] = df['diagnosis'].astype(str)

        train_df, temp_df = train_test_split(
            df, test_size=self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT, 
            stratify=df['diagnosis'], random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df, test_size=self.config.TEST_SPLIT / (self.config.VALIDATION_SPLIT + self.config.TEST_SPLIT),
            stratify=temp_df['diagnosis'], random_state=42
        )

        return train_df, val_df, test_df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    print("Data preprocessing module loaded successfully!")
