import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
img = cv.imread('test.jpg')
myVid = cv.VideoCapture('Ichigo.mp4')


detection = False
frameCounter = 0


success, imgVideo = myVid.read()

img = cv.resize(img, (1000,1000))
ht, wt, ct = img.shape
imgVideo = cv.resize(imgVideo, (wt, ht))

orb = cv.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(img, None)

# img = cv.drawKeypoints(img, kp1, None)

def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv.FILLED)
                cv.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

while True:


    success, imgWC  = cap.read()
    imgWC = cv.resize(imgWC, (wt, ht))
    imgAug = imgWC.copy()
    kp2, des2 = orb.detectAndCompute(imgWC, None)

    # imgWC = cv.drawKeypoints(imgWC, kp2, None)

    if detection == False:
        myVid.set(cv.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else: 
        if frameCounter == myVid.get(cv.CAP_PROP_FRAME_COUNT):
            myVid.set(cv.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv.resize(imgVideo, (wt, ht))

    bf = cv.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    imgFeatures = cv.drawMatches(img, kp1, imgWC, kp2, good, None, flags=2)

    if len(good) > 10:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0,0], [0,ht], [wt,ht], [wt,0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, matrix)
        img2 = cv.polylines(imgWC, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv.warpPerspective(imgVideo, matrix, (imgWC.shape[1], imgWC.shape[0]))        

        maskNew = np.zeros((imgWC.shape[0], imgWC.shape[1]), np.uint8)

        cv.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv.bitwise_not(maskNew)

        imgAug = cv.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv.bitwise_or(imgWarp, imgAug)

        imgStacked = stackImages(([imgWC, imgWarp, img2], [img, imgVideo, imgFeatures]), 0.5)
        
        imgStacked = cv.resize(imgStacked, (1000,1000))
        # cv.imshow('Image 2', img2)

        # cv.imshow('Image Warped', imgWarp)
        # cv.imshow('New Mask', maskNew)
        # cv.imshow('New Inversed Mask', maskInv)
        cv.imshow('Augmented Image', imgAug)
        cv.imshow('Stacked Image', imgStacked)

    # cv.imshow('Image', img)
    # cv.imshow('Video', imgVideo)
    # cv.imshow('Webcam', imgWC)
    # cv.imshow('Image Features', imgFeatures)
    

    cv.waitKey(1)
    frameCounter +=1

