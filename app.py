"""
Flask Web Application for Diabetic Retinopathy Prediction
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from config import Config

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
config = Config()
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable for model (load once at startup)
model = None
MODEL_PATH = 'models/saved_models/efficientnet_model_best.h5'

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def load_trained_model():
    """Load the trained model"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    return model

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, config.IMAGE_SIZE)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(image_path):
    """Make prediction on image"""
    model = load_trained_model()
    if model is None:
        return None, None, None

    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None, None, None

    predictions = model.predict(processed_image, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = config.CLASS_NAMES[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])

    # All predictions as dictionary
    all_predictions = {
        config.CLASS_NAMES[i]: float(predictions[0][i]) 
        for i in range(len(config.CLASS_NAMES))
    }

    return predicted_class, confidence, all_predictions

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        predicted_class, confidence, all_predictions = predict_image(filepath)

        if predicted_class is not None:
            return render_template('results.html',
                                 filename=filename,
                                 prediction=predicted_class,
                                 confidence=confidence,
                                 all_predictions=all_predictions)
        else:
            flash('Error processing image')
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload PNG, JPG, or JPEG images.')
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    predicted_class, confidence, all_predictions = predict_image(filepath)

    if predicted_class is not None:
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
    else:
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Diabetic Retinopathy Predictor Web Application...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
