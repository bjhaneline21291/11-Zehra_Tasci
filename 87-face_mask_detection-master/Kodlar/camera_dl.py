# USAGE
# python camera_dl.py --facemaskmodel mask.model

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# construct the argument parse and parse the arguments
arg = argparse.ArgumentParser()
arg.add_argument("-p", "--prototxt", required=False,
                default='./face_detector/deploy.prototxt',
                help="path to Caffe 'deploy' prototxt file")
arg.add_argument("-fm", "--facemodel", required=False,
                default='./face_detector/res10_300x300_ssd_iter_140000.caffemodel',
                help="path to Caffe pre-trained model")
arg.add_argument("-fmm", "--facemask_model", required=False,
                default='./mask_detector/mask.model',
                help="path to pre-trained model")
arg.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
arg.add_argument("-blur", "--faceblur_block",type=int, default=0,
                help="number of fuzzy blocks")
args = vars(arg.parse_args())


def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # return the pixelated blurred image
    return image


# load our human face serialized model from disk
print("[INFO] loading face model...")
faceNet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["facemodel"])
# net = cv2.dnn.readNetFromONNX(args["model"])
# load mask model
print("[INFO] loading mask model...")
maskNet = tf.keras.models.load_model(args["facemask_model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

labels_dict = {1: 'without_mask', 0: 'with_mask'}
color_dict = {1: (0, 0, 255), 0: (0, 255, 0)}

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=640, height=480)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    faceNet.setInput(blob)
    detections = faceNet.forward()
    face_count = detections.shape[2]
    # loop over the detections
    for i in range(face_count):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # compute mask model
        # fit ndim=4, ndim above is only 2
        size = 224
        face_img = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (size, size))

        # only normalization
        # normalized = face/255.0
        # reshaped = np.reshape(normalized, (1, size, size, 3))

        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        # reshape input size to fit the prediction
        # expected shape=(None, 224, 224, 3), found shape=(32, 224, 3)
        # reshaped = np.reshape(face, (1, size, size, 3))

        result = maskNet.predict(face, batch_size=32)

        # label = "Mask" if result[0][0] > result[0][1] else "No Mask"
        # color_dict = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = np.argmax(result, axis=1)[0]

        # compute face blur image
        # store the blurred face in the output image
        face_img = anonymize_face_pixelate(face_img,blocks=args["faceblur_block"])
        frame[startY:endY, startX:endX] = face_img

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      color_dict[label], 2)
        cv2.rectangle(frame, (startX, startY-40),
                      (endX, startY), color_dict[label], -1)
        # cv2.putText(frame, text, (startX, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(frame, labels_dict[label], (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # show the output frame
    cv2.imshow("LIVE", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
