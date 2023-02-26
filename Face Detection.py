#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

img = cv2.imread("testt.jpeg")
detections = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3,minSize=(3, 3) )

for face in detections:
        x,y,w,h = face
        img[y:y+h,x:x+w] = cv2.GaussianBlur(img[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),0)
        cv2.imshow("output",img)
      
cv2.waitKey(0)


# In[ ]:




