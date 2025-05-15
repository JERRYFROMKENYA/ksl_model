import cv2
import numpy as np
import tensorflow as tf
import json
import sys

# Check available cameras
def check_available_cameras():
    """Check which camera indices are available."""
    available_cameras = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="ksl_recognition_model/ksl_recognition_model.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit(1)

# Load class names
try:
    with open("ksl_recognition_model/class_names.json", "r") as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} class names successfully!")
except Exception as e:
    print(f"Error loading class names: {e}")
    sys.exit(1)

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Find available cameras
available_cameras = check_available_cameras()
if not available_cameras:
    print("No cameras detected. Please connect a camera and try again.")
    sys.exit(1)
else:
    print(f"Available camera indices: {available_cameras}")
    camera_index = available_cameras[0]  # Use the first available camera

# Open a video capture
print(f"Attempting to open camera at index {camera_index}...")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Failed to open camera at index {camera_index}. Please check your camera connection.")
    sys.exit(1)
else:
    print(f"Camera opened successfully at index {camera_index}!")

print("Press 'q' to quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame. Exiting...")
            break
        
        # Prepare the image for inference
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class
        predicted_class = np.argmax(output_data[0])
        
        # Ensure predicted_class is in range of class_names
        if predicted_class < len(class_names):
            confidence = output_data[0][predicted_class]
            prediction_text = f"{class_names[predicted_class]}: {confidence:.2f}"
        else:
            prediction_text = "Unknown class"
        
        # Display the result on the frame
        cv2.putText(frame, prediction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Kenyan Sign Language Recognition', frame)
        
        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred during execution: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")