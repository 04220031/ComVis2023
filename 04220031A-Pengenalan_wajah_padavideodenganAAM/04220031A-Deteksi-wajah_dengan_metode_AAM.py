# Import libraries
import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained drowsiness detection model
model = load_model('drowsiness_detection.h5')

# Define the function to detect drowsiness
def detect_drowsiness(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over the faces detected
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Resize the face to match the input size of the model
        roi_gray = cv2.resize(roi_gray, (24, 24))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        # Use the model to predict if the driver is drowsy or not
        preds = model.predict(roi_gray)
        
        # Get the prediction score
        score = preds[0][0]
        
        # Display the result
        if score < 0.5:
            # The driver is not drowsy
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Alert", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # The driver is drowsy
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Drowsy", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # If the frame was not read successfully, break out of the loop
    if not ret:
        break
    
    # Detect drowsiness in the frame
    frame = detect_drowsiness(frame)
    
    # Display the result
    cv2.imshow('Drowsiness Detection', frame)
    
    # If the user presses the 'q' key, break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()