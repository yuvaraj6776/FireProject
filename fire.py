#import the necessery packages


import numpy as np
import cv2

import time
import winsound
duration = 2000
freq = 440
#fire_detection.xml file & this code should be in the same folder while running the code

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while 1:
    
    #seperating frames from the video
    ret, img = cap.read()
    #implementing optical flow algorithm 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

   

    #implementing fire detection
    fire = fire_cascade.detectMultiScale(gray, 1.2, 5)

    #background subtraction
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
 
 
    output = cv2.bitwise_and(img, hsv, mask=mask)
    res = cv2.bitwise_and(img,img, mask= mask)
    for (x,y,w,h) in fire:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        print ('Fire is detected..!')
        winsound.Beep(freq, duration)
        
        
        
        
        time.sleep(0.2)
    
        
    cv2.imshow('img',img)
    cv2.imshow('back ground iamge',res)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

