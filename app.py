import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# 1. Load the pre-trained MobileNetV2 model
#    'weights='imagenet'' means it's trained on the ImageNet dataset
#    'include_top=True' means we include the classification layers
model = MobileNetV2(weights='imagenet', include_top=True)
print("MobileNetV2 model loaded successfully.")

# 2. Initialize webcam
cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Point your camera at common objects!")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Flip the frame horizontally for a mirror-like effect (optional, but common for webcams)
    frame = cv2.flip(frame, 1)

    # 3. Preprocess the frame for MobileNetV2
    # MobileNetV2 expects input images of size 224x224
    # We'll also convert BGR (OpenCV default) to RGB (Keras default)
    
    # Resize frame to 224x224
    img_resized = cv2.resize(frame, (224, 224))
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions to create a batch of 1 image (expected by Keras models)
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # Preprocess the input according to MobileNetV2 requirements
    # This scales pixel values to [-1, 1]
    preprocessed_img = preprocess_input(img_array)

    # 4. Predict the object
    predictions = model.predict(preprocessed_img)
    
    # Decode the predictions to get human-readable labels and confidence scores
    # top=1 means we only want the single best prediction
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    # Get the top prediction
    _, object_name, confidence = decoded_predictions[0]

    # 5. Display the results on the frame
    label = f"{object_name}: {confidence:.2f}"
    
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0) # Green color for text
    background_color = (0, 0, 0) # Black background for readability

    # Get text size to create a background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw a filled rectangle as a background for the text
    cv2.rectangle(frame, (0, 0), (text_width + 10, text_height + baseline + 10), background_color, -1)
    
    # Put the text on the frame
    cv2.putText(frame, label, (5, text_height + 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('AR Object Identifier', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()