from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError

from django.views.decorators import gzip
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import time
import imutils
import math
import numpy as np
import obj_detect as obj
class VideoCamera(object):
    
    def __init__(self,path):
        #self.video = cv2.VideoCapture(path)
        self.video = VideoStream(0).start()
        self.last_obj = []
        self.sh = 0
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
        
        
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
       
        self.pipl_cross = []
    def __del__(self):
        self.video.read()

    def get_frame(self):
        frame = self.video.read()
    
        fr = obj.run_models_recognition(frame)
       
        jpeg = cv2.imencode('.jpg',fr)[1].tostring()
        '''print(jpeg)
        j = jpeg.tobytes()
        print(j)'''
        return jpeg
    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def indexscreen(request): 
    try:
        template = "screens.html"
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")

@gzip.gzip_page
def dynamic_stream(request,num=0,stream_path="0"):
    
    stream_path = 'add your camera stream here that can rtsp or http'
    return StreamingHttpResponse(gen(VideoCamera(stream_path)),content_type="multipart/x-mixed-replace;boundary=frame")

