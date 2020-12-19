import sys 
from PyQt5.QtWidgets import (QWidget, QPushButton,
    QHBoxLayout, QVBoxLayout, QApplication, QFileDialog)
from PyQt5.QtCore import QCoreApplication
import cv2
import os
from Front import *
from PIL import Image
import glob
from fpdf import FPDF
import build
import obj_det
#import main_fase

video_path = '0'
textboxValue = "Kantrollzed@yandex.ru"

click = False

class MyWin(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.get_text)
        self.ui.pushButton_2.clicked.connect(self.showDialog)                     
        self.ui.pushButton_3.clicked.connect(self.run_models_Web)
        self.ui.pushButton_4.clicked.connect(self.exitprogram)
        self.ui.pushButton_5.clicked.connect(self.run_models_Video)
    
    def get_text(self):
        global click
        if not click:
            global textboxValue
            textboxValue = self.ui.lineEdit.text()
            print(textboxValue)
            self.ui.textBrowser.append("Confirmed: " + textboxValue + '\n')
            return textboxValue
        else: return
    
    def run_models_Web(self):
        global click
        if not click:
            path = '0'
            self.ui.textBrowser.append("Web-stream has been started...\n")
            obj_det.run_models_recognition(path)
            rez = obj_det.run_models_recognition(video_path)
            self.ui.textBrowser.append('in video be ' + str(rez)+ ' pipl')
        else: return
        
    def run_models_Video(self):
        global click
        if not click:
          
            self.ui.textBrowser.append("Registration activate please make some foto...\n")
            obj_det.fase_save(obj_det.reg())
            obj_det.build()
            obj_det.train()
            
            self.ui.textBrowser.append('you are registrate')
        else: return
        
    def showDialog(self):
        global click
        if not click:
            global video_path
            print("video_path: " + video_path)
            fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
            video_path = fname
            self.ui.textBrowser.setText(video_path)
            #tmp = ""
            #for i in range(len(video_path)):
            #   if video_path[i] == "/":
            #        tmp = tmp + "//"
            #    else:
            #        tmp += video_path[i]
            #video_path = tmp
            print("video_path: " + video_path)
            return video_path
        else: return
        
    def exitprogram(self): 
        sys.exit()      
        
        
        
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_())        
