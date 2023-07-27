import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

blank = np.zeros(img.shape[:2], dtype = 'uint8') # creating a blank image

b, g, r = cv.split(img) # splitting the image into its channels
blue = cv.merge([b, blank, blank]) # merging the blue channel with the blank image
green = cv.merge([blank, g, blank]) # merging the green channel with the blank image
red = cv.merge([blank, blank, r]) # merging the red channel with the blank image

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b, g, r]) # merging the channels back together
cv.imshow('Merged', merged)



cv.waitKey(0)

