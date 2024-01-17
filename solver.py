import cv2
import numpy as np

num=91
im = cv2.imread("input_shuffled.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 35, 255, 0)
contours, hierarch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# calculate areas
contours = [(cv2.contourArea(c),c) for c in contours]
# filter out very small contours
contours = [c for c in contours if c[0]>10]
to_remove = set()
for ci1 in range(len(contours)):
    for ci2 in range(ci1+1,len(contours)):
        # box is tuple x,y,w,h
        box1 = cv2.boundingRect(contours[ci1][1])
        box2 = cv2.boundingRect(contours[ci2][1])
        if not(box2[0]>box1[0]+box1[2] or box2[0]+box2[2]<box1[0] or box2[1]>box1[1]+box1[3] or box2[1]+box2[3]<box1[1]):
            if box1[0]<=box2[0] and box2[0]+box2[2]<=box1[0]+box1[2] and box1[1]<=box2[1] and box2[1]+box2[3]<=box1[1]+box1[3] or \
               box2[0]<=box1[0] and box1[0]+box1[2]<=box2[0]+box2[2] and box2[1]<=box1[1] and box1[1]+box1[3]<=box2[1]+box2[3]:
               overlap = 1
            else:
                overlap = 0
            for pix1 in contours[ci1][1]:
                if overlap:
                    break
                for pix2 in contours[ci2][1]:
                    if pix1[0][0] == pix2[0][0] and pix1[0][1] == pix2[0][1]:
                        overlap = 1
                        break
            if overlap:
                to_remove.add(ci1 if contours[ci1][0] < contours[ci2][0] else ci2)
contours = [c[1] for i,c in enumerate(contours) if i not in to_remove]
print(len(contours))
img = im.copy()
cv2.drawContours(img, contours, -1, (0,255,0), 2)
for c in contours:
    imc = im.copy()
    cv2.drawContours(imc, c, -1, (0,255,0), 2)
    cv2.imshow("1",imc)
    cv2.imshow("2",img)

    k = cv2.waitKey(0)
    if k == ord("q"):
        exit()