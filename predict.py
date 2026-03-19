import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("cnn_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128,128))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1,128,128,1)

    prediction = model.predict(reshaped)
    class_id = np.argmax(prediction)

    cv2.putText(frame, str(class_id), (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()