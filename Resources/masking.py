import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype = 'uint8') # creating a blank image
cv.imshow('Blank Image', blank)

circle = cv.circle(blank.copy(), (img.shape[1] // 2 + 100, img.shape[0] // 2), 100, 255, -1) # creating a mask
cv.imshow('Mask_circle', circle)

rectangle = cv.rectangle(blank.copy(), (img.shape[1] // 2 - 100, img.shape[0] // 2 - 100), (img.shape[1] // 2 + 100, img.shape[0] // 2 + 100), 255, -1) # creating a mask
cv.imshow('Mask_rec', rectangle)

mask = cv.bitwise_or(circle, rectangle) # masking the image
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask = mask) # masking the image
cv.imshow('Masked Image', masked)



cv.waitKey(0)