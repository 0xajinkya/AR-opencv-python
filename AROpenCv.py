import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
img = cv.imread('testImg.jpg')
myVid = cv.VideoCapture('Ichigo.mp4')

success, imgVideo = myVid.read()

img = cv.resize(img, (1000,1000))
ht, wt, ct = img.shape
imgVideo = cv.resize(imgVideo, (wt, ht))

orb = cv.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(img, None)

# img = cv.drawKeypoints(img, kp1, None)

while True:
    success, imgWC  = cap.read()
    imgWC = cv.resize(imgWC, (wt, ht))

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

    if len(good) > 10:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0,0], [0,ht], [wt,ht], [wt,0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, matrix)
        img2 = cv.polylines(imgWC, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv.warpPerspective(imgVideo, matrix, (imgWC.shape[1], imgWC.shape[0]))        

        cv.imshow('Image 2', img2)
        cv.imshow('Image Warped', imgWarp)

    cv.imshow('Image', img)
    cv.imshow('Video', imgVideo)
    cv.imshow('Webcam', imgWC)
    cv.imshow('Image Features', imgFeatures)
    

    cv.waitKey(0)

