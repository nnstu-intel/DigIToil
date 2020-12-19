# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:45:06 2019

@author: pitonhik
"""

import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
import Sanding_mail
import sqlBuild
import datetime
#import sql
import math
from imutils import paths
import numpy as np
import shutil
import Object_detection
import imutils
import pickle
import cv2
import os
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
    
    #////////////////////
    #///////////
    #///////////Object_detection
    #///////////
    #//////////////////
   
import dlib
def reg():
    print('введите свои данные')
    name = input('введите ваше имя на Английском: ')
    fam= input('введите вашу фамилию на Английском: ')
    day=str(datetime.date.today())
    city=input('введите город: ')
    mail=input('введите вашу электоронную почту: ')
    phone=input('введите ваш мобильный телефон: ')
    pos=input('введите вашу должность: ')
    
    true = input('ваше имя: ' + str(name) + ' ваша фамилия: ' + str(fam) + '  y/n/exit')
    if 'y' in true:
        sqlBuild.input_sql(name,fam,day,city,mail,phone,pos)
        return [name,fam]
    elif 'exit' in true:
        return False
    else:
        reg()
def delet():
     print('введите дааные удаляемого человека')
     name = input('введите  имя на Английском: ')
     fam = input('введите  фамилию на Английском: ')
     true = input('имя: ' + str(name) + ' фамилия: ' + str(fam) + '  y/n')
     if 'y' in true:
         try:
             sn=fam+'_'+name
             
             shutil.rmtree('dataset/'+sn)
             
         except:
              return 'not fail'
     elif 'exit' in true:
        return False
     else:
         delet()
def fase_save(name):
            
            print('программа снимает ваше лицо')
            sn=name[1]+'_'+name[0]
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
                    print('готово ' + str(foto)+ '/9 фото')
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
# When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()

def build():
 dataset = 'dataset'
 embeddings = 'output/embeddings.pickle'
 detetor = 'face_detection_model'

 em_model = 'openface_nn4.small2.v1.t7'
 # load our serialized face detector from disk
 print("[INFO] loading face detector...")
 protoPath = os.path.sep.join([detetor, "deploy.prototxt"])
 modelPath = os.path.sep.join([detetor,
    "res10_300x300_ssd_iter_140000.caffemodel"])
 detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

 # load our serialized face embedding model from disk
 print("[INFO] loading face recognizer...")
 embedder = cv2.dnn.readNetFromTorch(em_model)

 # grab the paths to the input images in our dataset
 print("[INFO] quantifying faces...")
 imagePaths = list(paths.list_images(dataset))

 # initialize our lists of extracted facial embeddings and
 # corresponding people names
 knownEmbeddings = []
 knownNames = []

 # initialize the total number of faces processed
 total = 0

 # loop over the image paths
 for (i, imagePath) in enumerate(imagePaths):
   try:
     # extract the person name from the image path
     print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
     name = imagePath.split(os.path.sep)[-2]

     # load the image, resize it to have a width of 600 pixels (while
     # maintaining the aspect ratio), and then grab the image
     # dimensions
     image = cv2.imread(imagePath)
     image = imutils.resize(image, width=600)
     (h, w) = image.shape[:2]

     # construct a blob from the image
     imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

     # apply OpenCV's deep learning-based face detector to localize
     # faces in the input image
     detector.setInput(imageBlob)
     detections = detector.forward()

     # ensure at least one face was found
     if len(detections) > 0:
         # we're making the assumption that each image has only ONE
         # face, so find the bounding box with the largest probability
         i = np.argmax(detections[0, 0, :, 2])
         confidence = detections[0, 0, i, 2]

         # ensure that the detection with the largest probability also
         # means our minimum probability test (thus helping filter out
         # weak detections)
         if confidence > 0:  
             # compute the (x, y)-coordinates of the bounding box for
             # the face
             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
             (startX, startY, endX, endY) = box.astype("int")

             # extract the face ROI and grab the ROI dimensions
             face = image[startY:endY, startX:endX]
             (fH, fW) = face.shape[:2]

             # ensure the face width and height are sufficiently large
             if fW < 20 or fH < 20:
                  continue

             # construct a blob for the face ROI, then pass the blob
             # through our face embedding model to obtain the 128-d
             # quantification of the face
             faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
             embedder.setInput(faceBlob)
             vec = embedder.forward()

             # add the name of the person + corresponding face
             # embedding to their respective lists
             knownNames.append(name)
             knownEmbeddings.append(vec.flatten())
             total += 1
   except:
      False
      print('image incorrect')
 # dump the facial embeddings + names to disk
 print("[INFO] serializing {} encodings...".format(total))
 data = {"embeddings": knownEmbeddings, "names": knownNames}
 f = open(embeddings, "wb")
 f.write(pickle.dumps(data))
 f.close()
def train():
 embeddings = 'output/embeddings.pickle'
 # load the face embeddings
 print("[INFO] loading face embeddings...")
 data = pickle.loads(open(embeddings, "rb").read())
 
 # encode the labels
 print("[INFO] encoding labels...")
 le = LabelEncoder()
 labels = le.fit_transform(data["names"])

 # train the model used to accept the 128-d embeddings of the face and
 # then produce the actual face recognition
 print("[INFO] training model...")
 recognizer = SVC(C=1.0, kernel="linear", probability=True)
 recognizer.fit(data["embeddings"], labels)
 rec = 'output/recognizer.pickle'
 # write the actual face recognition model to disk
 f = open(rec, "wb")
 f.write(pickle.dumps(recognizer))
 f.close()
 lee = 'output/le.pickle'
 # write the label encoder to disk
 f = open(lee, "wb")
 f.write(pickle.dumps(le))
 f.close()
detetor ='face_detection_model'
em_model = 'openface_nn4.small2.v1.t7'
rec = 'output/recognizer.pickle'
lee ='output/le.pickle'

protoPath = os.path.sep.join([detetor, "deploy.prototxt"])
modelPath = os.path.sep.join([detetor,
    "res10_300x300_ssd_iter_140000.caffemodel"])
detectorm = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


embedder = cv2.dnn.readNetFromTorch(em_model)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(rec, "rb").read())
le = pickle.loads(open(lee, "rb").read())
def detect(frame,detector):
 global le
 global recognizer
 global embedder

 # initialize the video stream, then allow the camera sensor to warm up



 # start the FPS throughput estimator


 # loop over frames from the video file stream
 if  True:
    # grab the frame from the threaded video stream
    
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    inframe = []
    obj ={}
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
       
        # filter out weak detections
        if confidence > 0.2:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # draw the bounding box of the face along with the
            # associated probability
            if proba > 0.2:
              text = "{}: {:.2f}%".format(name, proba * 100)
              y = startY - 10 if startY - 10 > 10 else startY + 10
              cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
              cv2.putText(frame, text, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
              obj['name']= name
              obj['ver'] = proba
              obj['kord'] = [startX, startY,endX, endY]
              
              inframe.append(obj)
            else:
              name = 'unknown'
              text = "{}: {:.2f}%".format(name, proba * 100)
              y = startY - 10 if startY - 10 > 10 else startY + 10
              cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
              cv2.putText(frame, text, (startX, y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
              obj['name']= name
              obj['ver'] = proba
              obj['kord'] = [startX, startY,endX, endY]
              
              inframe.append(obj)
           
    #cv2.imshow("q", frame)  
  
 return inframe
  

 

def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts

    coords = np.zeros((num, 2), dtype=dtype)

    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def get_centers(img, landmarks):
    # 线性回归
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255,0,0), 1) #画回归线
    cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# 瞳距
    scale = desired_dist / dist # 缩放比例
    angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face

def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0) #高斯模糊

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) #y方向sobel边缘检测
    sobel_y = cv2.convertScaleAbs(sobel_y) #转换回uint8类型
    #cv2.imshow('sobel_y',sobel_y)

    edgeness = sobel_y
    
    #Otsu二值化
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] #提取ROI
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])#计算评价值
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])#计算评价值
    measure = measure_1*0.3 + measure_2*0.7
    
    #cv2.imshow('roi_1',roi_1)
    #cv2.imshow('roi_2',roi_2)
    print(measure)
    

    if measure > 0.15:
        judge = True
    else:
        judge = False
    print(judge)
    return judge


predictor_path = "./data/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
def detect_glass(img):
 global detector
 global predictor

 if True:
    rez = []
    obg ={}
  
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for i, rect in enumerate(rects):
        obg ={}
        x_face = rect.left()
        y_face = rect.top()
        w_face = rect.right() - x_face
        h_face = rect.bottom() - y_face
        obg['kord']=[x_face,y_face,x_face+w_face,y_face+h_face]
       
        '''cv2.rectangle(img, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)'''
        
        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
        
        
        aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        #cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

        judge = judge_eyeglass(aligned_face)
        if judge == True:
            obg['glass'] = True
            #cv2.putText(img, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            obg['glass'] = False
            #cv2.putText(img, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)'''
        rez.append(obg)
        del(obg)
    return rez
    #cv2.imshow("Result", img)
last_f = []
def run_models_recognition(path):
    global detectorm
    global  last_f
    kolvo = 0
    path = str(path)
    PATH_TO_CKPT = 'modelFaster/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'modelFaster/Labelmap.pbtxt'
    NUM_CLASSES = 2
    files = ["C:\\tensorflow1\Done_Project_Faster_R-CNN_Caffe\Output\pdf"]   
    
    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    print('PATH_TO_LABELS='+PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.compat.v1.Session(graph=detection_graph)
    print('PATH_TO_CKPT='+PATH_TO_CKPT)
    
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    
    
    
    #////////////////////
    #///////////
    #///////////Real_time_object_detection
    #///////////
    #////////////////////
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    	"sofa", "train", "tvmonitor"]
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("modelCaffe/MobileNetSSD_deploy.prototxt.txt", "modelCaffe/MobileNetSSD_deploy.caffemodel")
    print("modelCaffe/MobileNetSSD_deploy.prototxt.txt")
    print("modelCaffe/MobileNetSSD_deploy.caffemodel")
    
    
    
    print("path: " + path)
    if path == '0':
        PATH_TO_VIDEO = 0
    else:
        PATH_TO_VIDEO = path
        
    # Open video file
    print(PATH_TO_VIDEO)
    video = cv2.VideoCapture(PATH_TO_VIDEO)
    
    w  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('(' + str(w) + ',' + str(h) + ')')
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Output\TestVideo1.avi', fourcc, 5, (w, h))
    
    iteration_Count = 0
    Col_pers = 0
    Sanded = False
    sh = 0
    while(video.isOpened()):
        sh+=1
        ret , frame = video.read()
       
        glass = detect_glass(frame)
       
        det =detect(frame,detectorm)
       
        centroids = []
        myrez = []
        Helmet = False
        Jacket = False
        Person = False
        

        
        
        #Caffe model///////////////////////////////////////////////////////////////////////////////////////////////////////////////   
        Resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(Resized, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    myrez.append([CLASSES[idx], startX, startY ,endX, endY])
        
        
        #Faster R-CNN//////////////////////////////////////////////////////////////////////////////////////////////////////////////
        frame_expanded = np.expand_dims(frame, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        # Draw the results of the detection (aka 'visulaize the results')
        myimg , rez= vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        if len(rez)>0:
            
            lang = len(rez)
            #print(lang)
            for i in range(lang):
                widt = int(rez[i][1]) + ((int(rez[i][2]) - int(rez[i][1]))/2)
                heidt = int(rez[i][3]) + ((int(rez[i][4]) - int(rez[i][3]))/2)
                centroids.append([rez[i][0], widt, heidt])
       
        helmet =[]
        jasket =[]
        for j in range(len(rez)):
            if rez[j][0]=='Chartreuse':
                jasket.append(rez[j])
            else:
                helmet.append(rez[j])
       
        leng = max(len(glass),len(det))
        lend = max(leng,len(myrez))
        main_mas_obg =[]
        main_obg={}
        
        for i in range(len(myrez)):
            main_obg={}
            main_obg['person'] = myrez[i][1:]
            main_obg['helmet']=False
            main_obg['jasket']=False
            main_obg['glass']=False
            main_obg['name'] = 'unknown'
            senter = [main_obg['person'][0] + (main_obg['person'][2]-main_obg['person'][0])/2,main_obg['person'][1] + (main_obg['person'][3]-main_obg['person'][1])/2]
            #print('sender')
            ##print('llllllllllllllll')
            #print(len(helmet))
            #print(len(jasket))
            for  j in range(len(helmet)):
                hsender = [helmet[j][1] + (helmet[j][3]-helmet[j][1])/2,helmet[j][2] + (helmet[j][4]-helmet[j][2])/2]
                #print('hsender')
                #print(hsender)
                lengt = math.sqrt((hsender[0]-senter[0])*(hsender[0]-senter[0])+(hsender[1]-senter[1])*(hsender[1]-senter[1]))
                #print(lengt)
                if lengt < 100:
                    main_obg['helmet']=True
            for  j in range(len(jasket)):
                hsender = [jasket[j][1] + (jasket[j][3]-jasket[j][1])/2,jasket[j][2] + (jasket[j][4]-jasket[j][2])/2]
                #print('aaa')
                #print(hsender)
                lengt = math.sqrt((hsender[0]-senter[0])*(hsender[0]-senter[0])+(hsender[1]-senter[1])*(hsender[1]-senter[1]))
                #print(lengt)
                if lengt < 100:
                    main_obg['jasket']=True
                    
            for j in range(len(glass)):
                hsender = [glass[j]['kord'][0] + (glass[j]['kord'][2]-glass[j]['kord'][0])/2,glass[j]['kord'][1] + (glass[j]['kord'][3]-glass[j]['kord'][1])/2]
                lengt = math.sqrt((hsender[0]-senter[0])*(hsender[0]-senter[0])+(hsender[1]-senter[1])*(hsender[1]-senter[1]))
                if lengt < 100:
                    main_obg['glass'] = glass[j]['glass']
                    break
            for j in range(len(det)):
                
                hsender = [det[j]['kord'][0] + (det[j]['kord'][2]-det[j]['kord'][0])/2,det[j]['kord'][1] + (det[j]['kord'][3]-det[j]['kord'][1])/2]
                lengt = math.sqrt((hsender[0]-senter[0])*(hsender[0]-senter[0])+(hsender[1]-senter[1])*(hsender[1]-senter[1]))
                if lengt < 100:
                    main_obg['name'] = det[j]['name']
                    break
            for j in range(len(det)):
                
                cv2.rectangle(frame, (det[j]['kord'][0], det[j]['kord'][1]), (det[j]['kord'][2], det[j]['kord'][3]),(0, 0, 255), 2)
                text = det[j]['name']
                cv2.putText(frame, text, (det[j]['kord'][0], det[j]['kord'][1]-15),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            main_mas_obg.append(main_obg)
            del(main_obg)
        #print('////////////////')
        print(main_mas_obg)
        if sh==1:
            kol = len(main_mas_obg)
        else:
            asd = len(main_mas_obg) - len(last_f)
            if asd > 0:
                kol+=asd
        line = 500
        for j in range(len(main_mas_obg)):
            if main_mas_obg[j]['helmet']==False:
                '''ЗАНОС В БД'''
                print('нет каски у ' + str(main_mas_obg[j]['name']))
                name_fam=main_mas_obg[j]['name'].split("_")
                sqlBuild.naruh_sql(name_fam[1],name_fam[0],'нет каски')
            if main_mas_obg[j]['jasket']==False:
                '''ЗАНОС В БД'''
                print('нет жилета у ' + str(main_mas_obg[j]['name']))
                name_fam=main_mas_obg[j]['name'].split("_")
                sqlBuild.naruh_sql(name_fam[1],name_fam[0],'нет жилета')
            if main_mas_obg[j]['glass']==False:
                '''ЗАНОС В БД'''
                print('нет очков у ' + str(main_mas_obg[j]['name']))
                name_fam=main_mas_obg[j]['name'].split("_")
                sqlBuild.naruh_sql(name_fam[1],name_fam[0],'нет очков')
            if main_mas_obg[j]['person'][2]>line:
                '''ЗАНОС В БД'''
                print('пересечение линии или зоны '+ str(main_mas_obg[j]['name']))
                name_fam=main_mas_obg[j]['name'].split("_")
                sqlBuild.naruh_sql(name_fam[1],name_fam[0],'пересечение линии')
        last_f = main_mas_obg
        out.write(frame)
        cv2.imshow('Object detector', frame)
        #print('--------------------')
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    
    video.release()
    out.release()
    cv2.destroyAllWindows()
    return kol
