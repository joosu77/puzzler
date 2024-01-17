import cv2
import numpy as np

num=91
im = cv2.imread("input_shuffled.png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 35, 255, 0)
contours, hierarch = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
areas = [(cv2.contourArea(c),c) for c in contours]
hist = {}
for a in areas:
    area, cont = a
    if area not in hist:
        hist[area]=[]
    hist[area].append(cont)
pieces = []
done=0
for k in sorted(hist.keys())[::-1]:
    for p in hist[k]:
        pieces.append(p)
        if len(pieces) == num:
            done = 1
            break
    if done:
        break

img = im.copy()
cv2.drawContours(img, pieces, -1, (0,255,0), 2)
for c in pieces[::-1]:
    imc = im.copy()
    cv2.drawContours(imc, c, -1, (0,255,0), 2)
    cv2.imshow("1",imc)
    cv2.imshow("2",img)

    k = cv2.waitKey(0)
    if k == ord("q"):
        exit()