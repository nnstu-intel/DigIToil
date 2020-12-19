# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'front.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import sys
#import build
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 396)
        MainWindow.setWindowTitle('Icon')
        MainWindow.setWindowIcon(QtGui.QIcon('каска.png'))
        MainWindow.setStyleSheet("background-image: url(:/newPrefix/cj.jpg);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(MainWindow)
        self.textBrowser.setGeometry(QtCore.QRect(220, 40, 256, 192))
        self.textBrowser.setStyleSheet("background-image: url(:/newPrefix/фон.png);")
        self.textBrowser.setObjectName("textBrowser")
        self.lineEdit = QtWidgets.QLineEdit(MainWindow)
        self.lineEdit.setGeometry(QtCore.QRect(230, 270, 241, 20))
        self.lineEdit.setStyleSheet("background-image: url(:/newPrefix/фон.png);")
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(230, 310, 171, 23))
        self.pushButton.setStyleSheet("background-image: url(:/newPrefix/фон.png);\n"
"font: 75 12pt \"Times New Roman\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 30, 141, 61))
        self.pushButton_2.setStyleSheet("background-image: url(:/newPrefix/фон.png);\n"
"font: 75 12pt \"Times New Roman\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 210, 141, 61))
        self.pushButton_3.setStyleSheet("background-image: url(:/newPrefix/фон.png);\n"
"font: 75 12pt \"Times New Roman\";")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 300, 141, 61))
        self.pushButton_4.setStyleSheet("background-image: url(:/newPrefix/фон.png);\n"
"font: 75 12pt \"Times New Roman\";")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(230, 240, 171, 20))
        self.label.setStyleSheet("font: 75  15pt \"Times New Roman\";\n"
"")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        
        self.pushButton_5 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 120, 141, 61))
        self.pushButton_5.setStyleSheet("background-image: url(:/newPrefix/фон.png);\n"
"font: 75 12pt \"Times New Roman\";")
        self.pushButton_5.setObjectName("pushButton_5")
        #MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 21))
        self.menubar.setObjectName("menubar")
        #MainWindow.setMenuBar(self.menubar)
        


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Construction"))
        self.pushButton.setText(_translate("MainWindow", "Подтвердить почту"))
        self.pushButton_2.setText(_translate("MainWindow", "Выбрать видео"))
        self.pushButton_3.setText(_translate("MainWindow", "Realtime"))
        self.pushButton_4.setText(_translate("MainWindow", "Выход"))
        self.label.setText(_translate("MainWindow", "E-mail получателя"))
        self.pushButton_5.setText(_translate("MainWindow", "Регистрация"))



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_MainWindow()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

