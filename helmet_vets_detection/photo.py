import cv2
import numpy as np
import time
def fase_save():
            
            
            
            
            faceCascade = cv2.CascadeClassifier('face_cascades/haarcascade_frontalface_default.xml')
            video_capture = cv2.VideoCapture(1)
            i =0
            
                
            foto = 0
                
            while foto < 8:
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
                    for (x, y, w, h) in faces:
                        
        #area = (x,y,x+w,y+h)
        #cv2.imwrite("image.jpg", frame)
#                        r = max(w, h) / 2
#                        centerx = x + w / 2
#                        centery = y + h / 2
#                        nx = int(centerx - r)
#                        ny = int(centery - r)
#                        nr = int(r * 2)
#                        faceimg = frame[ny:ny+nr, nx:nx+nr]
#                        lastimg = cv2.resize(faceimg, (150,150))
                        cv2.imwrite("faces/0000"+ str(i) +".jpg", frame)
                        foto +=1
                        i += 1
                        time.sleep(3)
    # Display the resulting frame
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
# When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()

fase_save()

