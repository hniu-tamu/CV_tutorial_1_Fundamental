import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype = 'uint8')
cv.imshow('Blank', blank)

# # 1. Paint the image a certain color
# blank[200:300, 300:400] = 0, 0, 255 # the color order is BGR
# cv.imshow('Red', blank)

# # 2. Draw a Rectangle
# cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//5), (0, 255, 0), thickness = -1, ) # cv.FILLED or -1 will fill the rectangle
# cv.imshow('Rectangle', blank)

# # 3. Draw a Circle
# cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness = -1) # center, radius, color, thickness
# cv.imshow('Circle', blank)

# # 4. Draw a Line
# cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness = 3)
# cv.imshow('Line', blank)

# 5. Write text
cv.putText(blank, 'Hello, my name is Haoyu', (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness = 2)
cv.imshow('Text', blank)
cv.waitKey(0)
