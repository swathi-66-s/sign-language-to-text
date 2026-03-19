import cv2
import numpy as np
from tensorflow.keras.models import load_model
labels = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',
    7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',
    13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W',
    23:'X', 24:'Y', 25:'Z'
}
model = load_model("cnn_model.h5")

cap = cv2.VideoCapture(0)
word = ""
count = 0
prev_letter = ""
while True:
    ret, frame = cap.read()
    roi = frame[100:300, 100:300]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128,128))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1,128,128,1)

    prediction = model.predict(reshaped)
    class_id = np.argmax(prediction)
    letter = labels[class_id]
    if letter == prev_letter:
    count += 1
else:
    count = 0

if count > 20:
    word += letter
    count = 0

prev_letter = letter
                cv2.putText(frame, letter, (50,50),
                 cv2.putText(frame, "Word: " + word, (50,100),           
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
