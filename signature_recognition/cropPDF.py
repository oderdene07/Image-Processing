from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils

images = convert_from_path("./testPDF.pdf")

img = np.array(images[-1])
orig = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
plt.imshow(gray,'gray')

edged = cv2.Canny(gray,80,200)
plt.imshow(edged,'gray')

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key= cv2.contourArea, reverse=True)[:1]
for c in cnts:
    peri = cv2.arcLength(c,True)
    apprx = cv2.approxPolyDP(c, 0.2*peri, True)
    x,y,w,h = cv2.boundingRect(apprx)
    cv2.rectangle(orig,(x,y),(x+w,y+h),(255,0,0),1)
plt.imshow(orig)

cropped_image = orig[y+3:y+h-3, x+3:x+w-5]
plt.imshow(cropped_image)

def remove_white_space(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    return image[y-5:y+h+5, x-18:x+w+20]

test_image = cv2.resize(cv2.cvtColor(remove_white_space(cropped_image.copy()),cv2.COLOR_BGR2GRAY),(212,97))
cv2.imwrite('./test_image.png', test_image)

