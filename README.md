# AI-Based Crop Disease Detection System

## Overview

This repository contains a Flask application prototype for
predicting a crop condition from an uploaded plant image.

The application uses TensorFlow/Keras to load a trained image
classification model and return a predicted class.

**Important:** A compatible trained model and the correct class
labels are required before the application can produce meaningful
crop disease predictions.

## Features

- Upload a plant image through a web form.
- Resize and normalize the image before prediction.
- Run inference with a TensorFlow/Keras model.
- Display the predicted class.

## Technologies

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow

## Project Structure

```text
.
├── README.md
├── app.py
├── requirements.txt
└── crop_disease_model.h5
```

The model file shown above is required by the current application
but has not been verified as present in the repository.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place a compatible trained model at the configured model path.
Update the class labels to match the model's output classes.

## Run

```bash
python app.py
```

Open the local address printed by Flask in your browser.

## Limitations

- A compatible trained model is required.
- The current class labels are placeholders.
- The model's training data, accuracy, and generalization have
  not been verified.
- Predictions should not be treated as confirmed agricultural
  diagnoses.
- The current application enables Flask debug mode and should
  not be deployed publicly in its present form.

## Future Improvements

- Document the dataset and training process.
- Add verified class names and evaluation metrics.
- Improve image validation and error handling.
- Add tests and sample images when licensing permits.

## Author

Islam Md Jubayer
GitHub: https://github.com/jubayeris
