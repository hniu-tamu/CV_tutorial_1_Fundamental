import cv2 as cv

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# simple thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) # threshold and thresh are the same because we are using binary thresholding

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) # if pixel value is greater than 150, it will be set to 0, else 255
cv.imshow('Simple Thresholded Inverse', thresh_inv) 


# Adaptive thresholding
# 11 is the block size, 3 is the constant that is subtracted from the mean
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 9) 
cv.imshow('Adaptive Thresholding', adaptive_thresh)


cv.waitKey(0)