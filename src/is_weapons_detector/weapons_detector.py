
import os
import cv2
import torch
import numpy as np





class WeaponsDetector:
    def __init__(self, model_conf): # inicializar a classe
    # device é GPU ou a CPU
    # vefica se tem GPU, se tiver device é GPU se não é CPU
        device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')
       
        # basicamente people_dectector carrga um modelo para detectar pessoas
        # esse modelo é do pessoal da yolo, vem de torch.hub
        # WeaponsDetector.people_detector
        
        self.people_detector = torch.hub.load('ultralytics/yolov5', "yolov5n",autoshape=True, pretrained=True, force_reload=True)
       
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
        #self.weapons_detector = torch.hub.load('ultralytics/yolov5',"custom", model_conf.weapon_model_file, force_reload=True), #estava assim antes
        self.weapons_detector = torch.hub.load('ultralytics/yolov5', "custom", model_conf.weapon_model_file, force_reload=True)
       
        # weapons_detector seria a classe do modelo, 2, armas de fogo e facas ?
        # WeaponsDetector.weapons_detector.classes
        self.weapons_detector.classes = [0] #only weapons 
       
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
        self.class_names = ["person"] + list(model_conf.weapon_class_names)
 

    def detect_weapons(self, image, max_size):

        """
        Detect weapons in the image

        Args:
            image (numpy.ndarray): frame to detect
            max_size (int): frame maximum size to detect
            save (bool): True to save frame with bounding box
        Returns:
            weapons(numpy.ndarray): result of prediction, index 0 to 3 are points of bounding box, index 4 confidence, index 5 index class
        """

        weapons = self.weapons_detector(image, max_size)

        weapons = weapons.xyxy[0] # prediction tensor
            
        return weapons


    

    
    def detect_people(self, image, max_size):

        """
        Detect people in the image

        Args:
            image (numpy.ndarray): frame to detect
            max_size (int): frame maximum size to detect
            save (bool): True to save frame with bounding box
        Returns:
            weapons(numpy.ndarray): result of prediction, index 0 to 3 are points of bounding box, index 4 confidence, index 5 index class
        """

        people = self.people_detector(image, max_size)
        
        people = people.xyxy[0] #.astype(int)
        return people


    def verification_gpu(self):

        print(f'cuda is {torch.cuda.is_available()}')
        

    