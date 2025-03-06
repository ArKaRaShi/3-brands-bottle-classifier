import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict

# Load your trained model
model = load_model("my_model.keras")

# Initialize confusion matrix as a defaultdict
confusion_matrix = defaultdict(lambda: defaultdict(int))
class_names = ["aura", "minere", "montfleur"]

def get_camera() -> cv2.VideoCapture:
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    print("No camera found.")
    return None  # Return None if no camera is found

cap = get_camera()
if cap is None:
    print("Unable to access any camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame: Resize, normalize, and add batch dimension
    img = cv2.resize(frame, (192, 192))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image

    # Make prediction using the model
    prediction = model.predict(img)

    # Get the class index with the highest probability and confidence score
    predicted_class = np.argmax(prediction, axis=-1)[0]
    confidence_score = prediction[0][predicted_class] * 100  # Convert to percentage

    # Map the predicted class index to a label
    label = class_names[predicted_class]
    
    # Update confusion matrix counter
    confusion_matrix[label][label] += 1

    # Format the confidence score text
    confidence_text = f"Confidence: {confidence_score:.2f}%"

    # Add the label and confidence score to the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, confidence_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display confusion matrix statistics
    y_offset = 130
    cv2.putText(frame, "Confusion Matrix:", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, class_name in enumerate(class_names):
        y_offset += 30
        count = confusion_matrix[class_name][class_name]
        stat_text = f"{class_name}: {count}"
        cv2.putText(frame, stat_text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Prediction", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
