import os
import sys
import cv2
import numpy as np
import time
import torch

from is_wire.core import Channel, Subscription, Message, Logger
from is_wire.rpc import ServiceProvider, LogInterceptor
from conf.msgs_pb2 import Image

from weapons_detector import WeaponsDetector
from is_utils import load_options, to_image, to_np, get_topic_id, bounding_box 


def place_images(output_image, images):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0:h, 0:w, :] = images[0]
    output_image[0:h, w:2 * w, :] = images[1]
    output_image[h:2 * h, 0:w, :] = images[2]
    output_image[h:2 * h, w:2 * w, :] = images[3]

# parte do rpc server, criar uma variavel logger
service_name = "GetImages.CameraGateway"
log = Logger(name=service_name)

op = load_options()

detector = WeaponsDetector(op.model)

channel = Channel(op.broker_uri)

 # log.info do ip das cameras que ele ta conetado
log.info('Connected to broker {}', op.broker_uri)

# Subscription para o cannal, e logger service_name
# Se escreve para o topico 'CameraGateway.*.Frame'
subscription = Subscription(channel=channel, name=service_name)

for c in op.cameras:
    subscription.subscribe('CameraGateway.{}.Frame'.format(c.id))

    
size = (2 * op.cameras[0].config.image.resolution.height,
        2 * op.cameras[0].config.image.resolution.width, 
        3)

puslish_name = 'Images.YOLOv5'
subscription_yolo = Subscription(channel=channel, name=puslish_name)

full_image = np.zeros(size, dtype=np.uint8)

gpu_activated = torch.cuda.is_available() # check if GPU is avaiable (bool)

# log info of GPu situation
if not gpu_activated:
    log.info('GPU is not enabled')
else:
    log.info('GPU is enabled')

# variebles
images_data = {}
n_sample = 0
for c in op.cameras:
    n_sample += 1
display_rate = 2
rod = False




while gpu_activated:

    msg = channel.consume()

    im = msg.unpack(Image)

    im_np = np.fromstring(im.data, dtype=np.uint8)

    camera_id = get_topic_id(msg.topic)

    # amazena as imagens(formato np) em suas posições
    images_data[camera_id] = im_np

    if len(images_data) == len(op.cameras):

                
        if n_sample % display_rate == 0:

            images = [0]*4

            for i, d in images_data.items():
                i = int(i) # id of the camera
                images[i] = cv2.imdecode(d, cv2.IMREAD_COLOR) # image numpy format
                    

            place_images(full_image, images) # reajusta o full_image para receber as imagens
            input_image_yolo = cv2.resize(full_image, (0, 0), fx=0.5, fy=0.5) # ajusted final image 

            infer_size = input_image_yolo.shape[1]
            infer_size = int(infer_size / 2) # minimum size for prediction

            detection = detector.detect_people(input_image_yolo, infer_size) # prediction
            

            # draw bounding box in the frame
            output_image_yolo = input_image_yolo
            output_image_yolo = bounding_box(output_image_yolo, detections=detection, class_names=detector.class_names, infer_conf=detector.people_detector.conf)

            output_image = to_image(output_image_yolo) # image for publish (Image)

            msg2publish = Message(content=output_image, reply_to=subscription_yolo)  # Message for publish (Message)

            channel.publish(msg2publish, topic='YoloPrediction.Frame') # Publish the frame with the yolo prediction and bounding box

            

            

            




            