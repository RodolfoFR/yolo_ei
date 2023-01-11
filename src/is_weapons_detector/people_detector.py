import torch

class PeopleDetector:

    def __init__(self, model_conf):


        device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')

        self.detector = torch.hub.load('ultralytics/yolov5', "yolov5n",autoshape=True, pretrained=True, force_reload=True)

        self.detector.classes = [0]

        self.detector.conf = model_conf.people_nms_conf
       
        # joga o modelo dectector para o device
        self.detector = self.detector.to(device)

     # recebe model_conf.people_infer_img_size (argurmento da classe), acho que é tamanho para bbox detectar, valor minimo aceitavél
        #  WeaponsDetector. people_infer_img_size
        self.people_infer_img_size = model_conf.people_infer_img_size

        self.class_name = ["person"]

    
    def detect(self, image, max_size):

        """
        Detect people in the image

        Args:
            image (numpy.ndarray): frame to detect
            max_size (int): frame maximum size to detect
        Returns:
            weapons(numpy.ndarray): result of prediction, index 0 to 3 are points of bounding box, index 4 confidence, index 5 index class
        """

        people = self.detector(image, max_size)
        
        people = people.xyxy[0].cpu().numpy() #.astype(int)
        return people