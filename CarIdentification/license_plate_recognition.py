# Импорт все библиотек
import os
import cv2
import numpy as np
import sys
import json
import matplotlib.image as mpimg
import imutils
from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync
from ISU_vision import car


class LP_Detector():
    def __init__(self, img):
        # путь к корню
        self.NOMEROFF_NET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

        # путь к MaskRCNN (если она не в корне)
        self.MASK_RCNN_DIR = os.path.join(self.NOMEROFF_NET_DIR, 'Mask_RCNN')
        self.MASK_RCNN_LOG_DIR = os.path.join(self.NOMEROFF_NET_DIR, 'logs')

        sys.path.append(self.NOMEROFF_NET_DIR)

        

        # Initialize npdetector with default configuration file.
        self.nnet = Detector(self.MASK_RCNN_DIR, self.MASK_RCNN_LOG_DIR)
        self.nnet.loadModel("latest")

        self.rectDetector = RectDetector()

        self.optionsDetector = OptionsDetector()
        self.optionsDetector.load("latest")

        # Активируем детектор текста
        self.textDetector = TextDetector({
            "eu_ua_2004_2015": {
                "for_regions": ["eu_ua_2015", "eu_ua_2004"],
                "model_path": "latest"
            },
            "eu": {
                "for_regions": ["eu", "eu_ua_1995"],
                "model_path": "latest"
            },
            "ru": {
                "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
                "model_path": "latest" 
            },
            "kz": {
                "for_regions": ["kz"],
                "model_path": "latest"
            },
            "ge": {
                "for_regions": ["ge"],
                "model_path": "latest"
            }
        })

        self.path = path
        self.image = cv2.imread(path)

    def process(self):

        print("\n++++++++++")
        print("START RECOGNIZING")
        print("++++++++++\n")

        # Путь к папке с фотками
        #img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images/bbb.jpg')

        max_img_w = 1600

        img = self.img
        start_img = img.copy()

        # изменяем размер фото для увеличения скорости
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_w_r = 1
        img_h_r = 1
        if img_w > max_img_w:
            resized_img = cv2.resize(img, (max_img_w, int(max_img_w/img_w*img_h)))
            img_w_r = img_w/max_img_w
            img_h_r = img_h/(max_img_w/img_w*img_h)
        else:
            resized_img = img

        NP = self.nnet.detect([resized_img])

        # Генерируем маску
        cv_img_masks = filters.cv_img_mask(NP)

        # Находим координаты бокса номера
        arrPoints = self.rectDetector.detect(cv_img_masks, outboundHeightOffset=0, fixGeometry=True, fixRectangleAngle=10)
        arrPoints[..., 1:2] = arrPoints[..., 1:2]*img_h_r
        arrPoints[..., 0:1] = arrPoints[..., 0:1]*img_w_r

        # Рисуем все боксы
        filters.draw_box(img, arrPoints, (0, 255, 0), 3)

        # вырезаем эти зоны для дальнейшего анализа
        zones = self.rectDetector.get_cv_zonesBGR(img, arrPoints)

        # находим стандарт номера, соответствующий стране (номер будет отформатирован под него)
        regionIds, stateIds, countLines = self.optionsDetector.predict(zones)
        regionNames = self.optionsDetector.getRegionLabels(regionIds)
        print("\nСтраны:" ,regionNames)

        # находим текст и форматируем по стандарту страны
        textArr = self.textDetector.predict(zones, regionNames, countLines)
        textArr = textPostprocessing(textArr, regionNames)
        print("Обнаруженный текст", textArr)

        # выводим результат в окне
        start_img = imutils.resize(start_img, width = 600, height = 400)
        img = imutils.resize(img, width = 600, height = 400)
        return img, textArr


model_detector = car()
img, info = model_detector.info("images/1.jpg")
detector = LP_Detector(img)
img, text = detector.process()

print(text)       

