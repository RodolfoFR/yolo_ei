# import os
# import cv2
# import torch
# import yolov5
# import numpy as np
# from is_utils import crop_image, increase_bbox
# from msgs_pb2 import ObjectAnnotation



# class WeaponsDetector:
#     def __init__(self, model_conf):
#         device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')
#         self.people_detector = torch.hub.load('ultralytics/yolov5', model_conf.people_detection_model, pretrained=True)
#         self.people_detector.classes = [0]
#         self.people_detector.conf = model_conf.people_nms_conf
#         self.people_detector = self.people_detector.to(device)
#         self.people_infer_img_size = model_conf.people_infer_img_size
#         self.increase_bbox_percent = model_conf.increase_image_percent/100.0


#         self.weapons_detector = yolov5.load(model_conf.weapon_model_file)
#         self.weapons_detector.classes = [0, 2] #only weapons and knifes
#         self.weapons_detector.conf = model_conf.weapon_nms_conf
#         self.weapons_detector = self.weapons_detector.to(device)
#         self.weapon_infer_img_size = model_conf.weapon_infer_img_size
     
#         self.class_names = ["pessoa"] + list(model_conf.weapon_class_names)
 

#     def __translate_detections(self,dets,rois):
#         detections = np.empty((0,6))
#         for det,roi in zip(dets,rois):
#             if len(det)>0:
#                 det[...,[0,2]] += roi[0]
#                 det[...,[1,3]] += roi[1]
#                 detections = np.concatenate([detections,det])
#         return detections
        

#     def __detect_people(self, image):
#         people = self.people_detector(image, self.people_infer_img_size)
#         people = np.array(people.xyxy[0].cpu().numpy()).astype(int)
#         return people

#     def __detect_weapons(self, image, people):
#         weapons = np.empty((0,6))
#         if len(people)>0:
#             people = increase_bbox(image.shape[:2],people, borders = self.increase_bbox_percent)
#             people_images = crop_image(image,people)
#             weapons = self.weapons_detector(people_images, self.weapon_infer_img_size)
#             weapons = [weapon.cpu().numpy() for weapon in weapons.xyxy]
#             weapons = self.__translate_detections(weapons,people)
#             weapons = weapons.reshape(-1,6)
#             weapons[...,-1] += 1
#         return weapons


#     def detect(self, image):       
#         people = self.__detect_people(image)
#         weapons = self.__detect_weapons(image,people)
        
#         # print(people.shape,weapons.shape)
#         detections = people
#         if len(weapons)>0:
#             detections = np.concatenate([people,weapons])
#         return detections
import os
import cv2
#import yolov5
import torch
import numpy as np
# before was .is_utils in from to import crop_image, increase_bbox, original file
from is_utils import crop_image, increase_bbox
# before was .conf.msgs_pb2 in from to import ObjectAnnotation, original file
from conf.msgs_pb2 import ObjectAnnotation




class WeaponsDetector:
    def __init__(self, model_conf): # inicializar a classe
    # device é GPU ou a CPU
    # vefica se tem GPU, se tiver device é GPU se não é CPU
        device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')
       
        # basicamente people_dectector carrga um modelo para detectar pessoas
        # esse modelo é do pessoal da yolo, vem de torch.hub
        # WeaponsDetector.people_detector
        self.people_detector = torch.hub.load('ultralytics/yolov5', "yolov5s",autoshape=True, pretrained=True, force_reload=True)
       
        # people_detector só tem uma classe que deve se pessoa
        # WeaponsDetector.people_detector.classes
        self.people_detector.classes = [0]
       
        # recebe model_conf.people_nms_conf (argurmento da classe), acho que é valor minimo de precisão para detectar pessoa
        # WeaponsDetector.people_detector.conf
        self.people_detector.conf = model_conf.people_nms_conf
       
        # joga o modelo people-dectector para o device
        self.people_detector = self.people_detector.to(device)
       
        # recebe model_conf.people_infer_img_size (argurmento da classe), acho que é tamanho para bbox detectar, valor minimo aceitavél
        #  WeaponsDetector. people_infer_img_size
        self.people_infer_img_size = model_conf.people_infer_img_size
       
        # recebe model_conf.increase_image_percent (argurmento da classe), acho que é p tamanho do bbox nas imagens
        # WeaponsDetector.increase_bbox_percent
        self.increase_bbox_percent = model_conf.increase_image_percent/100.0

        # weapons_detector vai ser o modelo para prever armas
        # modelo custumizado, recebe model_conf.weapon_model_file (argurmento da classe), que são os arquivos dos pesos
        # acho que ele pega do proprio git deles o arquivos de pesos, usa o modelo 'custom' da yolo
        # WeaponsDetector.weapons_detector
        # self.weapons_detector = torch.hub.load('ultralytics/yolov5',"custom", model_conf.weapon_model_file, force_reload=True), estava assim antes
        self.weapons_detector = torch.hub.load('ultralytics/yolov5', "yolov5s", autoshape=True, pretrained=True, force_reload=True)
       
        # weapons_detector seria a classe do modelo, 2, armas de fogo e facas ?
        # WeaponsDetector.weapons_detector.classes
        self.weapons_detector.classes = [0, 1] #only weapons and knifes
       
        # recebe model_conf.weapon_nms_conf (argurmento da classe), acho que é valor minimo de precisão para detectar arma
        # WeaponsDetector.weapons_detector.conf
        self.weapons_detector.conf = model_conf.weapon_nms_conf
       
        # joga o modelo weapons_detector para o device
        self.weapons_detector = self.weapons_detector.to(device)
       
        # recebe model_conf.people_infer_img_size (argurmento da classe), acho que é tamanho para bbox detectar, valor minimo aceitavél
        # WeaponsDetector.weapon_infer_img_size
        self.weapon_infer_img_size = model_conf.people_infer_img_size
     
      # Recebe model_conf.weapon_class_names (argurmento da classe), classe de WeaponsDetector em string
      # list(model_conf.weapon_class_names) tem ["Arma", "Pessoa Armada"], mas weapons_detector.classes são armas de fogo e facas ????
      # WeaponsDetector.class_names
        self.class_names = ["pessoa"] + list(model_conf.weapon_class_names)
 

    def __translate_detections(self,dets,rois):
        detections = np.empty((0,6))
        for det,roi in zip(dets,rois):
            if len(det)>0:
                det[...,[0,2]] += roi[0]
                det[...,[1,3]] += roi[1]
                detections = np.concatenate([detections,det])
        return detections
       
    
    def detect_people(self, image, infer_size):

        people = self.people_detector(image, infer_size)
        people = np.array(people.xyxy[0].cpu().numpy()) #.astype(int)
        return people

    def __detect_weapons(self, image):
        if isinstance(image,list):
            image = [cv2.GaussianBlur(image,(7,7),1)]
        else:
            image = cv2.GaussianBlur(image,(7,7),1)
        # prever se a pessoa esta com arma
        people = self.weapons_detector(image, self.people_infer_img_size)
        # coordenadas na imagem (de aonde está a pessoa com arma), a acuracia e a classe (em indice)
        people = np.array(people.xyxy[0].cpu().numpy()).astype(int)
        return people # returna em numpy


    def detect(self, image):
       
        weapons = self.__detect_weapons(image)
       
        # print(people.shape,weapons.shape)
        # detections = np.ar
        # if len(weapons)>0:
        # detections = np.concatenate([people,weapons])
        weapons = weapons.reshape(-1,6)
        weapons[...,-1] += 1
        return weapons

    def verification_gpu(self):

        print(f'cuda is {torch.cuda.is_available()}')
        

