from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model (this should be the .h5 file of your trained skin disease model)
model = load_model('telemedicine/model/skin_disease_classifier.h5')

# Define the size to which the image should be resized (224x224)
IMG_SIZE = (224, 224)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Load and preprocess the image
    image = Image.open(file)
    image = image.resize(IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)

    # You can map predicted_class to skin disease names
    class_names = ["acne", "eczema","fungal","psoriasis", "ringworm"]
    result = class_names[predicted_class[0]]

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
