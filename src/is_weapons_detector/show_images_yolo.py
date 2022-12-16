import os
import sys
import cv2
import numpy as np
import time
import torch 

import dateutil.parser as dp

# before was .conf.msgs_pb2 in from to import Image, original file
from conf.msgs_pb2 import Image
#from is_msgs.image_pb2 import Image
from is_wire.core import Logger, Subscription, Message, Tracer, Channel

from weapons_detector import WeaponsDetector
from is_utils import load_options, create_exporter, get_topic_id,  to_image, to_np, annotate_image, bounding_box, random_hex_id, save_video
from stream_channel import StreamChannel





"""def span_duration_ms(span):
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0"""

# congifurar os tamanhos tas imagens, para plotar elas juntas
def place_images(output_image, images):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0:h, 0:w, :] = images[0]
    output_image[0:h, w:2 * w, :] = images[1]
    output_image[h:2 * h, 0:w, :] = images[2]
    output_image[h:2 * h, w:2 * w, :] = images[3]



# parte do rpc server, criar uma variavel logger
service_name = "WeaponsDetector.Detect"
log = Logger(name=service_name)
   
# load_options(), uma funcao,  vem de is_utils.py
# vai fazer um log.info do que vai ter dentro de op
# basicamente op vai ser da classe WeaponDetectorOptions que vem de conf/options.proto
# nessa classe: str .broker_uri = ip da camera (ou espaco), str .zipkin_uri = ip do zipkin_uri, YoloModel .model = parametro do modelo
op = load_options()
   
# detector vai ser da classe WeaponsDetector que vem de weapons_detector.py, Rede para classificar
# que tem como parametro op.model, que sao os parametro do modelo (vai ter os pesos)
detector = WeaponsDetector(op.model)


# StreamChannel eh uma classe que vem de stream_channel.py
# eh parecido com Channel do is_wire.core
# Mas ele eh um canal, consume mensagem, se ele esperar demais, ele para, ou quando para a conexao
channel = Channel(op.broker_uri)
   
# log.info do ip das cameras que ele ta conetado
log.info('Connected to broker {}', op.broker_uri)

# create_exporter, eh uma funcao de ia_utils.py, parametros: o nome do logger (rpc server) e ip do zipkin
# exporter eh alguma coisa que envolver o zipkin
#exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)
   
# Subscription para o cannal, e logger service_name
# Se escreve para o topico 'CameraGateway.*.Frame'
subscription = Subscription(channel=channel, name=service_name)
# subscription no topico do ip das 4 cameras
for c in op.cameras:
    subscription.subscribe('CameraGateway.{}.Frame'.format(c.id))

# get the size of the images from the camera 
size = (2 * op.cameras[0].config.image.resolution.height,
        2 * op.cameras[0].config.image.resolution.width, 
        3)

# variables 

# empty array, but in the proper format
full_image = np.zeros(size, dtype=np.uint8)
#display_image  = np.zeros(size, dtype=np.uint8)



images_data = {}
n_sample = 0
for c in op.cameras:
    n_sample += 1
display_rate = 2

first = 0
rod = False

# to determine fps    
start_time = 0
end_time = 1

video_name = random_hex_id()
id_frame_save = 0

# If true recording the frames (bool)
recording = True

# If True activate the detector (bool)
detector_activated = False

gpu_activated = torch.cuda.is_available() # check if GPU is avaiable (bool)


# log info of GPu situation
if not gpu_activated:
    log.info('GPU is not enabled')
else:
    log.info('GPU is enabled')


# only runs when GPU is on     
while gpu_activated:

    start_time = time.time()
   
    # parametro return_dropped=True serve alem de receber a messagem receber o dropped
    # consume a mensagem  
    msg = channel.consume()
       
    # nao sei o msg.extract_tracing(),
    #tracer = Tracer(exporter, span_context=msg.extract_tracing())
    #span = tracer.start_span(name='detection_and_render')
    #detection_span = None

    #with tracer.span(name='unpack'):

    # trata a mensagem recebido como imagem
    im = msg.unpack(Image)
    # transforma a imagem em numpy
    im_np = np.fromstring(im.data, dtype=np.uint8)
            

    # pega o topic da mensagem e dividi ela
    # retorna so o numero do id (0, 1, 2, 3), se estiver correta a formatacao
    camera_id = get_topic_id(msg.topic)

    # amazena as imagens(formato np) em suas posicoes
    images_data[camera_id] = im_np
    

                
    # check if there are the same amount of images and cameras
    if len(images_data) == len(op.cameras): 

                   
        if n_sample % display_rate == 0:

            images = [0, 0, 0, 0]

            for i, d in images_data.items():
                i = int(i) # id of the camera
                images[i] = cv2.imdecode(d, cv2.IMREAD_COLOR) # image numpy format
                    
            display_image  = np.zeros(size, dtype=np.uint8)
            place_images(display_image, images) # adjusts the full_image to receive the images
  
            

            max_size = display_image.shape[1]
            max_size = int(max_size / 2) # minimum size for prediction

            

            if detector_activated:

                detection_weapons = detector.detect_weapons(display_image, max_size) # prediction weapons
                # draw bounding box in the frame according weapon detector prediction
                display_image = bounding_box(display_image, detections=detection_weapons, class_names=detector.class_names, infer_conf=detector.weapons_detector.conf, weapon=True)

                detection_people = detector.detect_people(display_image, max_size) # prediction people
                # draw bounding box in the frame according weapon detector prediction
                display_image = bounding_box(display_image, detections=detection_people, class_names=detector.class_names, infer_conf=detector.people_detector.conf)
               
                            

            end_time = time.time() # end 

            fps = int( 1 / (end_time - start_time) ) # fps of video 

            # write the fps in the frames
            display_image = cv2.putText(display_image, f'fps: {fps}', (5, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(250,250,250), thickness=2, lineType=cv2.LINE_AA)    
                                       
            display_image = cv2.resize(display_image, (0, 0), fx=0.5, fy=0.5) # ajusted final image

            if recording:
                # save frames
                cv2.imwrite(f'/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/Videos/frames/{video_name}-{id_frame_save}.jpg', display_image)
                id_frame_save += 1
            
            

            cv2.imshow('YOLO', display_image) # display images
            key = cv2.waitKey(1)

           
                    

            if key == 'q':
                break

                    

            
