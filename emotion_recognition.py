from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from collections import deque   # for smoothing predictions

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Create a buffer to store last N predictions
buffer_size = 10
pred_buffer = deque(maxlen=buffer_size)

while True:
    # Grab the webcamera's image.
    ret, frame = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    pred_buffer.append(index)

    # Take most common prediction in buffer (majority vote)
    stable_index = max(set(pred_buffer), key=pred_buffer.count)
    class_name = class_names[stable_index].strip()
    confidence_score = prediction[0][stable_index]

    # Print prediction and confidence score
    print("Class:", class_name, "Confidence:", str(np.round(confidence_score * 100))[:-2], "%")

    # Show the webcam image with prediction overlay
    cv2.putText(frame, f"{class_name} ({confidence_score*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Image", frame)

    # ESC to quit
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
