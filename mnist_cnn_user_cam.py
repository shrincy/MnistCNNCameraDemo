# Author: Rogine
# Date: 2017-09-13

from keras.models import load_model
import numpy as np
import cv2 as cv


# Image size for CNN input
img_rows, img_cols = 28, 28
# Roi range param
# Please verify this range to suit your view for processing
x1, x2, y1, y2 = 200, 440, 120, 360
# Binary threshold value
# Please verify this value and check the binary image
thres  = 60
# Dilate process kernel
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


# Load CNN trained model
# Please verify your path here
path = './CNN.h5'
model = load_model(path)

# Create uvc camera video capture
video_capture = cv.VideoCapture(0)

while True:
    # Get one frame for processing
    res, frame = video_capture.read()
    if res == False:
        print('Capture read frame failed!')
        continue

    # Get roi range
    roiImg = frame[y1:y2, x1:x2]

    # Binary process
    grayImg = cv.cvtColor(roiImg, cv.COLOR_BGR2GRAY)
    _, binImg = cv.threshold(grayImg, thres, 255, cv.THRESH_BINARY_INV)
    cv.imshow('binImg', binImg)

    # Dilate process to avoid object breakpoint
    binImg = cv.dilate(binImg, kernel)
    cv.imshow('processed binImg', binImg)

    # input data process
    input = cv.resize(binImg, (img_rows, img_cols))
    input = input.reshape((1, img_rows, img_cols, 1))

    # Predict and get result
    pred = model.predict(input, 1)
    result = np.argmax(pred, axis=1)

    # Draw result
    cv.putText(roiImg, str(result), (150, 200),
               cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,0), 2)
    # Draw roi range
    cv.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
    # Show result
    cv.imshow('result', frame)

    # Press ESC to exit
    if cv.waitKey(10) == 27:
        break

video_capture.release()
cv.destroyAllWindows()