import os
import cv2
import torch
import numpy as np


class Simple:

    def __init__(self, model_conf):
        
        device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')


        self.people_detector = torch.hub.load('ultralytics/yolov5', "yolov5s",autoshape=True, pretrained=True, force_reload=True)

        self.people_detector.classes = [0]

        self.people_detector.conf = model_conf.people_nms_conf

        self.people_detector = self.people_detector.to(device)

        self.class_names = ["pessoa"]

    def detect_people(self, image, infer_size):

        people = self.people_detector(image, infer_size)
        people = np.array(people.xyxy[0].cpu().numpy()) #.astype(int)
        return people


