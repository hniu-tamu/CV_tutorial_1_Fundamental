import cv2 as cv

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# averaging
average = cv.blur(img, (7, 7)) # blur is the same as average
cv.imshow('Average Blur', average)


# Gaussian Blur, which is less blurry than average because it takes the pixels around the center and gives them more weight
gauss = cv.GaussianBlur(img, (7, 7), 0) # 0 is the standard deviation
cv.imshow('Gaussian Blur', gauss)

# Median Blur, which is less blurry than Gaussian Blur because it takes the median of the pixels around the center
median = cv.medianBlur(img, 7) # 7 is the kernel size
cv.imshow('Median Blur', median)

# Bilateral Blur, which is the least blurry because it keeps the edges intact
bilateral = cv.bilateralFilter(img, 10, 35, 35) # 10 is the diameter, 35 is the sigma color, 25 is the sigma space, 
# sigma color is the color difference, sigma space is the space difference
cv.imshow('Bilateral Blur', bilateral)


cv.waitKey(0)





