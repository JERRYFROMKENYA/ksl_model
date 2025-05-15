
# Kenyan Sign Language Recognition Model

This package contains a TensorFlow Lite model for recognizing Kenyan Sign Language (KSL) and translating it to English.

## Files:
- ksl_recognition_model.tflite: The trained TFLite model
- class_names.json: JSON file with the mapping of class indices to sign meanings

## Model Information:
- Input shape: 224x224x3 (RGB image normalized to [0,1])
- Output: Class probabilities corresponding to different KSL signs

## Usage:
1. Load the model using TensorFlow Lite
2. Preprocess the input image to 224x224 RGB
3. Normalize the pixel values to [0,1]
4. Run inference
5. Get the class with the highest probability
6. Map the class index to the sign meaning using class_names.json
