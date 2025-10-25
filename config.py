"""
Configuration file for Diabetic Retinopathy Predictor
"""

import os

class Config:
    # Data paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')
    TEST_DIR = os.path.join(DATA_DIR, 'test_images')
    LABELS_DIR = os.path.join(DATA_DIR, 'labels')

    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Class names for DR severity levels
    CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    NUM_CLASSES = 5

    # Model save path
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'saved_models')

    # Data augmentation parameters
    ROTATION_RANGE = 30
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True

    # Training parameters
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

    # Web app configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'web_app', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
