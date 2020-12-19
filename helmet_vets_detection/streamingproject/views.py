from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError,HttpResponseRedirect
from django.shortcuts import redirect
from django.views.decorators import gzip
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import time
import imutils
import math
import numpy as np
import obj_detect as obj
import mimetypes
import os
myauth = False
line = 200
def get_line():
    global line
    
    return line
def line_edit(r):
    global line
    line = r
    
def get_info(mas,box):
    global id_s
    d = 1000
    imas = 0
    info = []
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
        li = get_line()
        li = int(li)
        fr = obj.stream(frame,li)
        cv2.line(fr,(0,li),(2000,li ),(0,255,0),thickness=2)
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
   global myauth
  
   
   try:
      if myauth:
       
        template = "screens.html"
        return render(request,template)
      else:
        template = "auth.html"
        return render(request,template)
   except HttpResponseServerError:
        print("aborted")
def download(request):
         file_name ='othet.txt'
         fp = open(file_name, "rb");
         response = HttpResponse(fp.read());
         fp.close();
         file_type = mimetypes.guess_type(file_name);
         if file_type is None:
             file_type = 'application/octet-stream';
         response['Content-Type'] = file_type
         response['Content-Length'] = str(os.stat(file_name).st_size);
         response['Content-Disposition'] = "attachment; filename=othet.txt";
 
         return response;
def auth(request):
    global myauth
    print('-------------------------------------------')
    print(request)
    req = str(request).split('/')
    name = req[2]
    pas = req[3]
    pas = pas[:-2]
    print(str(req))
    print('-------------------------------------------')
    print(name)
    print(pas)
    if name == 'tima' and pas =='qwerty':
        myauth = True
        print('auth succesful')
        return HttpResponseRedirect('/stream/screen')
def del_auth(request):
    global myauth
    myauth = False
    return redirect('/stream/screen')
def changeline(request):
    global line
    print('-------------------------------------------')
    print(request)
    line = str(request).split('/')
    line = int(line[2])
    print(str(line))
    print('-------------------------------------------')
    return HttpResponseRedirect('/stream/screen')
@gzip.gzip_page
def dynamic_stream(request,num=0,stream_path="0"):
    
    stream_path = 'add your camera stream here that can rtsp or http'
    return StreamingHttpResponse(gen(VideoCamera(stream_path)),content_type="multipart/x-mixed-replace;boundary=frame")

