# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:18:23 2020

@author: timka
"""

import cv2

import torch
import torchvision
from torchvision import models, transforms


'''
пример запуск c = car()
c.info('1.jpg')

под фрем переписывается путем удаления первой же строчки в def info()
фрем должен иметь такой же формат

парамет уверенности по умолчанию 0,7

изначльно классы  ['B_sedan','B_suv','c']
потом в зависимости от типа
['LADA_PRIORA_B','MAZDA_3_B','VОLКSWАGЕN_РОLО_B','КIА_RIО_B','НУUNDАI_SОLАRIS_B']
['RЕNАULТ_DUSТЕR_B','TOYOTA_RАV4_B','VОLКSWАGЕN_TIGUAN_B']
['KAMAZ_ALLKAMAZ_C','SCANIA_ALLSCANIA_C','VOLVO_ALLVOLVO_C']
или unknown

пример выхода {'class': 'sedan', 'name': 'LADA_PRIORA_B', 'ver': 0.99623126}
класс достаточно создать 1 раз
'''


class car(object):
    def __init__(self):
        model_car_class = 'ResNet50_Opened_vs_Closed_v2_B_C.pth'
        model_C = 'ResNet50_C.pth'
        model_B1 = 'ResNet50_B1.pth'
        model_B2 = 'ResNet50_B2.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_car = ['B_sedan', 'B_suv', 'C']
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 3)
        self.model.load_state_dict(torch.load(model_car_class))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.class_B1 = ['LADA_PRIORA_B', 'MAZDA_3_B', 'VОLКSWАGЕN_РОLО_B', 'КIА_RIО_B', 'НУUNDАI_SОLАRIS_B']
        self.modl1 = models.resnet50(pretrained=True)
        self.modl1.fc = torch.nn.Linear(self.modl1.fc.in_features, 5)
        self.modl1.load_state_dict(torch.load(model_B1))
        self.modl1 = self.modl1.to(self.device)
        self.modl1.eval()

        self.class_B2 = ['RЕNАULТ_DUSТЕR_B', 'TOYOTA_RАV4_B', 'VОLКSWАGЕN_TIGUAN_B']
        self.modl2 = models.resnet50(pretrained=True)
        self.modl2.fc = torch.nn.Linear(self.modl2.fc.in_features, 3)
        self.modl2.load_state_dict(torch.load(model_B2))
        self.modl2 = self.modl2.to(self.device)
        self.modl2.eval()

        self.class_C = ['KAMAZ_ALLKAMAZ_C', 'SCANIA_ALLSCANIA_C', 'VOLVO_ALLVOLVO_C']
        self.modl3 = models.resnet50(pretrained=True)
        self.modl3.fc = torch.nn.Linear(self.modl3.fc.in_features, 3)
        self.modl3.load_state_dict(torch.load(model_C))
        self.modl3 = self.modl3.to(self.device)
        self.modl3.eval()

    def get_predictions(self, model, input_image):
        img = input_image
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image = img
        img = transforms.ToTensor().__call__(img)
        img = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).__call__(img)
        img = img.unsqueeze_(0).to(self.device)
        img_dataset = torch.utils.data.TensorDataset(img)
        img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=1)
        for img in img_loader:
            imag = img[0]
            with torch.set_grad_enabled(False):
                preds = model(imag)
            prediction = torch.nn.functional.softmax(preds, dim=1).data.cpu().numpy()

            return image, prediction

    def info_class(self, image):
        img, predictions = self.get_predictions(self.model, image)
        pred = predictions[0].argmax()
        BoxClass = self.class_car[pred]
        return BoxClass

    def info(self, img):
        image = cv2.imread((img), cv2.IMREAD_UNCHANGED)
        par = self.info_class(image)
        if par == 'B_sedan':
            img, predictions = self.get_predictions(self.modl1, image)
            pred = predictions[0].argmax()
            ver = predictions[0][pred]
            BoxClass = self.class_B1[pred]
            Class = BoxClass
            Class = Class.split('_')
            ver = float(ver)
            if ver > 0.7:
                return {"class": 'B', "brand": Class[0], "model": Class[1], "probability": ver}
            else:
                return {"class": 'B', "brand": "unknown", "model": "unknown", "probability": ver}

        elif par == 'B_suv':
            img, predictions = self.get_predictions(self.modl2, image)
            pred = predictions[0].argmax()
            ver = predictions[0][pred]
            BoxClass = self.class_B2[pred]
            Class = BoxClass
            Class = Class.split('_')
            ver = float(ver)
            if ver > 0.7:
                return {"class": 'B', "brand": Class[0], "model": Class[1], "probability": ver}
            else:
                return {"class": 'B', "brand": "unknown", "model": "unknown", "probability": ver}
        else:
            img, predictions = self.get_predictions(self.modl3, image)
            pred = predictions[0].argmax()
            ver = predictions[0][pred]
            BoxClass = self.class_C[pred]
            Class = BoxClass
            Class = Class.split('_')
            ver = float(ver)
            if ver > 0.7:
                return {"class": 'C', "brand": Class[0], "model": Class[1], "probability": ver}
            else:
                return {"class": 'C', "brand": "unknown", "model": "unknown", "probability": ver}
