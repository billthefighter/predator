#This file exists to test the process shown here: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
#This file also exists to show how computer vision detects tom cruise

import numpy as np
import cv2

   
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

poop = cv2.imread('tomcruise.jpg')
poop2 = cv2.imread('tomcruise2.jpg')
poop3 = cv2.imread('tomcruise3.jpg')

def facedetect(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return(img)


cv2.imshow('farts',facedetect(poop))
cv2.imshow('farts2',facedetect(poop2))
cv2.imshow('farts3',facedetect(poop3))
cv2.waitKey(0)
cv2.destroyAllWindows()