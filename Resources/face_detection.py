import cv2 as cv


img = cv.imread('Resources/Photos/group 1.jpg')
cv.imshow('Group of people', img)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('Resources/haar_face.xml') # haar_cascade is a face detector
# scaleFactor is the amount by which the image is scaled down, 
# minNeighbors is the number of neighbors each rectangle should have to be called a face
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print(f'Number of faces found = {len(faces_rect)}') # len(faces_rect) is the number of faces found
print(faces_rect)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) # (0, 255, 0) is the color of the rectangle, thickness is the thickness of the rectangle
cv.imshow('Detected Faces', img)





cv.waitKey(0)