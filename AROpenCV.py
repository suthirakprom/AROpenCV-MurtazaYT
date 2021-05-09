import cv2
import numpy as np
from PIL import Image

image = Image.open("D:\Programs\PycharmProjects\AROpenCV\gfirst.png")
image = image.resize((500,500), Image.ANTIALIAS)
image.save(fp="newimage.png")


cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('newimage.png')
myVid = cv2.VideoCapture('jujutsu.jpg')

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)

while True:
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print(len(good))
    imgFeature = cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)

    if len(good) > 20:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

    cv2.imshow("maskNew", maskNew)
    cv2.imshow("ImageWarp", imgWarp)
    cv2.imshow("ImageFeature", imgFeature)
    cv2.imshow("ImgTarget", imgTarget)
    cv2.imshow("MyImage", imgVideo)
    cv2.imshow("Webcam", imgWebcam)
    cv2.waitKey(0)