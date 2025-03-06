import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("my_model.keras")

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
    img = img / 255.0  # Normalize the image (assuming your model was trained with this normalization)

    # Make prediction using the model
    prediction = model.predict(img)

    # Get the class index with the highest probability
    predicted_class = np.argmax(prediction, axis=-1)[0]

    # Map the predicted class index to a label
    class_names = ["aura", "minere", "montfleur"]  # Replace this with your actual class names
    label = class_names[predicted_class]

    # Add the label to the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the predicted label
    cv2.imshow("Real-Time Prediction", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
