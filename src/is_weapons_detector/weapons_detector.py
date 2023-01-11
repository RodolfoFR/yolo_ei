import torch






class WeaponsDetector:
    def __init__(self, model_conf): # inicializar a classe
    # device é GPU ou a CPU
    # vefica se tem GPU, se tiver device é GPU se não é CPU
        device = torch.device(f'cuda:{model_conf.gpu}' if (model_conf.gpu>=0 and torch.cuda.is_available()) else 'cpu')
       
        

        # weapons_detector vai ser o modelo para prever armas
        # modelo custumizado, recebe model_conf.weapon_model_file (argurmento da classe), que são os arquivos dos pesos
        # acho que ele pega do proprio git deles o arquivos de pesos, usa o modelo 'custom' da yolo
        # WeaponsDetector.weapons_detector
        #self.weapons_detector = torch.hub.load('ultralytics/yolov5',"custom", model_conf.weapon_model_file, force_reload=True), #estava assim antes
        self.detector = torch.hub.load('ultralytics/yolov5', "custom", model_conf.weapon_model_file, force_reload=True)
       
        # weapons_detector seria a classe do modelo, 2, armas de fogo e facas ?
        # WeaponsDetector.weapons_detector.classes
        self.detector.classes = [0] #only weapons 
       
        # recebe model_conf.weapon_nms_conf (argurmento da classe), acho que é valor minimo de precisão para detectar arma
        # WeaponsDetector.weapons_detector.conf
        self.detector.conf = model_conf.weapon_nms_conf
       
        # joga o modelo weapons_detector para o device
        
        self.detector = self.detector.to(device)
       
        # recebe model_conf.people_infer_img_size (argurmento da classe), acho que é tamanho para bbox detectar, valor minimo aceitavél
        # WeaponsDetector.weapon_infer_img_size
        self.weapon_infer_img_size = model_conf.people_infer_img_size
     
      # Recebe model_conf.weapon_class_names (argurmento da classe), classe de WeaponsDetector em string
      # list(model_conf.weapon_class_names) tem ["Arma", "Pessoa Armada"], mas weapons_detector.classes são armas de fogo e facas ????
      # WeaponsDetector.class_names
        self.class_name = ["weapon"]

    def detect(self, image, max_size):

        """
        Detect weapons in the image

        Args:
            image (numpy.ndarray): frame to detect
            max_size (int): frame maximum size to detect
            
        Returns:
            weapons(numpy.ndarray): result of prediction, index 0 to 3 are points of bounding box, index 4 confidence, index 5 index class
        """

        weapons = self.detector(image, max_size)

        weapons = weapons.xyxy[0].cpu().numpy() # prediction tensor
            
        return weapons


    

    
    

   
        

    