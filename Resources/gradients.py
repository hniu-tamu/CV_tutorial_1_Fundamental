import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian

lap = cv.Laplacian(gray, cv.CV_64F) # cv.CV_64F is the data type
lap = np.uint8(np.absolute(lap)) # convert to unsigned 8-bit integer
cv.imshow('Laplacian', lap)


# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0) # 1 is the order of the derivative in x direction
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1) # 1 is the order of the derivative in y direction
combined_sobel = cv.bitwise_or(sobelx, sobely)


cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

# Canny
canny = cv.Canny(gray, 150, 175) # 150 and 175 are the threshold values
cv.imshow('Canny', canny)

cv.waitKey(0)