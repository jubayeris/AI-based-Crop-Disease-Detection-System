# AI-based-Crop-Disease-Detection-System
from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model (make sure the model is available)
model = tf.keras.models.load_model('crop_disease_model.h5')

# Class labels (update these with your actual class names)
class_labels = ['Healthy', 'Disease A', 'Disease B', 'Disease C', 'Disease D']

# Simple HTML form for file upload
html_form = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Disease Detection</title>
</head>
<body>
    <h1>Upload a Plant Image for Disease Prediction</h1>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
'''

# Home route that displays the upload form
@app.route('/')
def home():
    return render_template_string(html_form)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    image = Image.open(file)
    image = image.resize((150, 150))  # Resize the image to match the model input size
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]  # Get the predicted class

    return f"Predicted Disease: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
