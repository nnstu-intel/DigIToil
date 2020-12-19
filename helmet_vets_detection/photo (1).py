import cv2
import numpy as np
import time
import os
name = input("имя")
fam = input("fam")
def fase_save(name,surname):
            
           
            sn=surname+'_'+name
            os.mkdir('dataset/'+sn)
            os.getcwd()
            faceCascade = cv2.CascadeClassifier('face_detection_model/haarcascade_frontalface_default.xml')
            video_capture = cv2.VideoCapture(0)
            i =0
            
               
            foto = 0
            fram = 0 
            while foto < 9:
    # Capture frame-by-frame
                    
                    ret, frame = video_capture.read()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(100, 100),
                            flags=cv2.CASCADE_SCALE_IMAGE
                            )
    # Draw a rectangle around the faces
                    print(fram)
                    if fram > 0 and fram%80 ==0:
                            for (x, y, w, h) in faces:
                             cv2.imwrite("dataset/"+sn+"/0000"+ str(i) +".jpg", frame)
                             foto +=1
                             i += 1
                      
    # Display the resulting frame
                    fram+=1
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
# When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()

fase_save(name,fam)

