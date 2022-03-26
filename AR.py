# this is my first clone project Using OpenCv In Ogmented Reality !!
import numpy as np

import cv2

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) ## To open the camera with OpenCv

imgTarget = cv2.imread('TergerImage.jpg')
myVid = cv2.VideoCapture('ObjectVideo.mp4')

detection = False
frameCounter = 0



success, imgVideo = myVid.read() # get the images in each frame of the video


heightTarget ,widthTarget ,cTarget = imgTarget.shape # get the shape of the target image
imgVideo = cv2.resize(imgVideo ,(widthTarget,heightTarget)) # resize the shape of our video to fit the target image

orb = cv2.ORB_create(nfeatures=1000)
kp1, dec1 = orb.detectAndCompute(imgTarget, None)
#imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)


while True :


    success, imgWebcam = cap.read()

    imgAug = imgWebcam.copy()
    kp2, dec2 = orb.detectAndCompute(imgWebcam, None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    if detection == False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_POS_FRAMES):
            myVid.set(cv2.CAP_PROP_POS_FRAMES ,0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (widthTarget, heightTarget))

    def stackImages(imgArray, scale, lables=[]):
        sizeW = imgArray[0][0].shape[1]
        sizeH = imgArray[0][0].shape[0]
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
                hor_con[x] = np.concatenate(imgArray[x])
            ver = np.vstack(hor)
            ver_con = np.concatenate(hor)
        else:
            for x in range(0, rows):
                imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            hor_con = np.concatenate(imgArray)
            ver = hor
        if len(lables) != 0:
            eachImgWidth = int(ver.shape[1] / cols)
            eachImgHeight = int(ver.shape[0] / rows)
            print(eachImgHeight)
            for d in range(0, rows):
                for c in range(0, cols):
                    cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
        return ver


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(dec1 ,dec2 ,k = 2)
    good = []

    for m,n in matches :
        if m.distance < 0.75 *n.distance :
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget ,kp1 ,imgWebcam ,kp2 ,good ,None ,flags = 2)

    if len(good) >= 0 :
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        destPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts ,destPts ,cv2.RANSAC ,5)

        print(matrix)

        pts = np.float32([[0,0], [0,heightTarget] ,[widthTarget ,heightTarget], [widthTarget ,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        img2 = cv2.polylines(imgWebcam ,[np.int32(dst)] ,True ,(255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(imgVideo ,matrix ,(imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew ,[np.int32(dst)] ,(255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug ,imgAug ,mask= maskInv)

        imgAug = cv2.bitwise_or(imgWarp ,imgAug)

        imgStacked = stackImages(([imgWebcam,imgVideo ,imgTarget],[imgFeatures ,imgWarp ,imgAug]),0.5)




    cv2.imshow('Final Result ',imgAug)
    cv2.imshow('aumented video ! ',imgWarp)
    #cv2.imshow('yaaa cadrihaa ',img2)
    #cv2.imshow('imageFeatures ',imgFeatures)
    #cv2.imshow('imageTarget',imgTarget)
    #cv2.imshow('Video Goal',imgVideo)
    #cv2.imshow('WebCam image',imgWebcam)
    cv2.imshow('image stacked',imgStacked)

    cv2.waitKey(1)
    frameCounter +=1