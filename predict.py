import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("plant_disease_model.h5")

classes = ["Early_blight","Late_blight","Healthy"]

img = cv2.imread("test_leaf.jpg")
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.reshape(img,(1,128,128,3))

prediction = model.predict(img)

print("Predicted disease:",classes[np.argmax(prediction)])
