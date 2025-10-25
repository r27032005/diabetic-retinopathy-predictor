"""
CNN Model Architecture for Diabetic Retinopathy Classification
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import Config

class DRModel:
    def __init__(self):
        self.config = Config()
        self.model = None

    def create_custom_cnn(self, input_shape=(224, 224, 3)):
        """Create a custom CNN architecture for DR classification"""
        inputs = Input(shape=input_shape)

        # First Convolutional Block
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Second Convolutional Block
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Third Convolutional Block
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Fourth Convolutional Block
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)

        # Fully Connected Layers
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output layer
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_efficientnet_model(self, input_shape=(224, 224, 3)):
        """Create transfer learning model using EfficientNetB3"""
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        base_model.trainable = False

        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.config.NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model, base_model

    def compile_model(self, model, learning_rate=None):
        """Compile the model with optimizer and loss function"""
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE

        optimizer = Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    def get_callbacks(self, model_name='dr_model'):
        """Get training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{self.config.MODEL_SAVE_PATH}/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        return callbacks

if __name__ == "__main__":
    dr_model = DRModel()
    print("Creating custom CNN model...")
    custom_model = dr_model.create_custom_cnn()
    print(f"Custom model created with {custom_model.count_params():,} parameters")
