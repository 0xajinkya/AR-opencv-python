import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
img = cv.imread('Cards.webp')
myVid = cv.VideoCapture('Ichigo.mp4')

success, imgVideo = myVid.read()
ht, wt, ct = img.shape

imgVideo = cv.resize(imgVideo, (wt, ht))

orb = cv.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(img, None)

# img = cv.drawKeypoints(img, kp1, None)

while True:
    success, imgWC  = cap.read()
    kp2, des2 = orb.detectAndCompute(imgWC, None)

    # imgWC = cv.drawKeypoints(imgWC, kp2, None)

    bf = cv.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    imgFeatures = cv.drawMatches(img, kp1, imgWC, kp2, good, None, flags=2)
    cv.imshow('Image', img)
    cv.imshow('Video', imgVideo)
    cv.imshow('Webcam', imgWC)
    cv.imshow('Image Features', imgFeatures)

    cv.waitKey(0)

