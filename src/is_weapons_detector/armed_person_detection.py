import os
import cv2
import torch
import yolov5
import numpy as np
from .is_utils import crop_image, increase_bbox
from .conf.msgs_pb2 import ObjectAnnotation



class WeaponsDetector:
    def __init__(self, model_conf):
        device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')
        self.people_detector = torch.hub.load('ultralytics/yolov5', model_conf.people_detection_model, pretrained=True)
        self.people_detector.classes = [0]
        self.people_detector.conf = model_conf.people_nms_conf
        self.people_detector = self.people_detector.to(device)
        self.people_infer_img_size = model_conf.people_infer_img_size
        self.increase_bbox_percent = model_conf.increase_image_percent/100.0


        self.weapons_detector = yolov5.load(model_conf.weapon_model_file)
        self.weapons_detector.classes = [0, 1] #only weapons and knifes
        self.weapons_detector.conf = model_conf.weapon_nms_conf
        self.weapons_detector = self.weapons_detector.to(device)
        self.weapon_infer_img_size = model_conf.weapon_infer_img_size
     
        self.class_names = ["pessoa"] + list(model_conf.weapon_class_names)
 

    def __translate_detections(self,dets,rois):
        detections = np.empty((0,6))
        for det,roi in zip(dets,rois):
            if len(det)>0:
                det[...,[0,2]] += roi[0]
                det[...,[1,3]] += roi[1]
                detections = np.concatenate([detections,det])
        return detections
        

    def __detect_people(self, image):
        people = self.people_detector(image, self.people_infer_img_size)
        people = np.array(people.xyxy[0].cpu().numpy()).astype(int)
        return people

    def __detect_weapons(self, image):
        people = self.weapons_detector(image, self.people_infer_img_size)
        people = np.array(people.xyxy[0].cpu().numpy()).astype(int)
        return people


    def detect(self, image):
        
        people = self.__detect_people(image)
        weapons = self.__detect_weapons(image,people)
        
        # print(people.shape,weapons.shape)
        detections = people
        if len(weapons)>0:
            detections = np.concatenate([people,weapons])
        return detections
    
