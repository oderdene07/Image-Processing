import cv2
import numpy as np
import random as rd

image = cv2.imread('1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
gray = gray // 255 * 255
isLetter = False
letterList = []
savingIndex = 0
for i in range(width):
    if not isLetter and 0 in gray[:, i]:
        isLetter = True
        savingIndex = i
    if isLetter and not 0 in gray[:, i]:
        isLetter = False 
        letterList.append((savingIndex, i))

colims = []
for lett in letterList:
    colims.append(gray[:, lett[0]:lett[1]])

for im in colims:
    print(im.shape)
    for i in range(im.shape[0]):
        if not isLetter and 0 in im[i]:
            isLetter = True
            savingIndex = i
        if isLetter and not 0 in im[i]:
            isLetter = False
            imag = im[savingIndex:i]
            name = rd.randint(1, 1000)
            cv2.imwrite("images/" + str(name) + ".jpg", imag)

print(letterList) 
