import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("MaskDetectionModel.h5")
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    img_cropped = frame
    for (x, y, w, h) in faces:
        img_cropped = frame[y:y+h,x:x+w]
    img_cropped = cv2.resize(img_cropped, (128, 128)) 
    img_cropped = np.expand_dims(img_cropped, axis=0)
    X= np.array(img_cropped)/255
    if(model.predict(X)>0.99):
        text = "Vous portez une bavette"
        color = (0,255,0)
    else:
        text = "Vous ne portez pas une bavette"
        color = (0,0,255)
    cv2.putText(frame, text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_4)
    cv2.imshow("MaskDetector", frame)
	# To exit press Esc Key.
    key = cv2.waitKey(1)
    if key == 27:
        capture.release()
        cv2.destroyAllWindows()
        break
