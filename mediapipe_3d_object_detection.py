#pip install mediapipe
#pip install opencv-contrib-python

import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

objectron = mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.4,
                            min_tracking_confidence=0.70,
                            model_name='Cup')
                            
while cap.isOpened():
    success, image = cap.read()

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            
            mp_drawing.draw_landmarks(image, 
                                      detected_object.landmarks_2d, 
                                      mp_objectron.BOX_CONNECTIONS)
            
            mp_drawing.draw_axis(image, 
                                 detected_object.rotation,
                                 detected_object.translation)

    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()