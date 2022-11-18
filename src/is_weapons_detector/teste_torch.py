#import torch
#import yolov5
import numpy as np
import cv2
from is_wire.core import Channel,Subscription,Message
from is_msgs.image_pb2 import Image
from is_utils import load_options, bounding_box
#from weapons_detector import WeaponsDetector


def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


broker_uri = "amqp://guest:guest@localhost:5672"
camera_id = 3 # selecionar camera
channel = Channel(broker_uri)
subscription = Subscription(channel=channel)
subscription.subscribe(topic='CameraGateway.{}.Frame'.format(camera_id))
#op = load_options()
#detector = WeaponsDetector(op.model)


while True:

    msg = channel.consume()  
    im = msg.unpack(Image)
    frame = to_np(im)

    #detection = detector.detect_people(frame, frame.shape[1])
    #frame = bounding_box(frame, detections=detection, class_names=detector.class_names, infer_conf=detector.people_detector.conf)
    

    cv2.imshow('test', frame)
    key = cv2.waitKey(1)
        
    if key == ord('q'):
        break
    elif key == ord('d'):
        print('enabled')



#       xmin       ymin        xmax        ymax      confidence  class  name
#0   81.855957   1.319438  184.182846  109.907242    0.850786      0  person
#1    0.000000  23.669359   62.045010  143.517242    0.831257      0  person
#2  136.652161   6.607765  272.116699  179.696915    0.787901      0  person
#3   43.493324  87.764862   60.556637  105.465462    0.402087     41     cup




