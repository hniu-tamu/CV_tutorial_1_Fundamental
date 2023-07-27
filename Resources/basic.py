import cv2 as cv

img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Boston', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT) # (3, 3) is the kernel size
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(blur, 125, 175) # 125 and 175 are the threshold values
cv.imshow('Canny Edges', canny)

# Dilating the image
dilated = cv.dilate(canny, (7, 7), iterations = 3) # iterations is the thickness of the edges
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7, 7), iterations = 3)
cv.imshow('Eroded', eroded)

# resize
resized = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)