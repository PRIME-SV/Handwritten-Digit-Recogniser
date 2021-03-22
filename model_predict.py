import requests
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
# loading the saved model that we trained.
model = load_model('model-010.model')

# loading the image and processing it
image_org = cv2.imread("digits_1.jpg")
# blurring the image to remove unwanted discontinuity in the image
image_org = cv2.GaussianBlur(image_org, (5, 5), 0)
image_org = cv2.resize(image_org, (800, 500))                       # resizing
# converting to grayscale
image = cv2.cvtColor(image_org.copy(), cv2.COLOR_BGR2GRAY)
# thresholding to remove noises
ret, thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('result', thresh)

# function to extraxt digits from image
cntr, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cntr:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image_org, (x, y), (x+w, y+h), (255, 0, 0), 2)

    img = thresh[y:y+h, x:x+w]

    # we need to resize it and then apply proper padding to match image to that of dataset
    img = cv2.resize(img, (18, 18))
    padded_img = np.pad(img, ((5, 5), (5, 5)), constant_values=0)

    final_img = padded_img.reshape(1, 28, 28, 1)
    result = str(int(model.predict_classes(final_img)))
    print(result)
    cv2.putText(image_org, result, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    #cv2.imshow('result', final_img.reshape(28, 28))
to_save = np.asarray(image_org)
cv2.imwrite("Output.jpg", to_save)
cv2.imshow('result', image_org)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Made by SDV