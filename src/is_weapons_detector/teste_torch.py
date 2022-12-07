import torch
#import yolov5
import numpy as np
import cv2
import time
from is_wire.core import Channel,Subscription, Message, Logger
from is_msgs.image_pb2 import Image
from is_utils import load_options, bounding_box, to_image
from weapons_detector import WeaponsDetector
#from simple_class import Simple


def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


# variables

op = load_options() # infos as broker uri, zipkin uri, paremeters of model and cameras

detector = WeaponsDetector(op.model) 

broker_uri = "amqp://guest:guest@localhost:5672"
camera_id = 2 # select camera
channel = Channel(broker_uri)
subscription = Subscription(channel=channel)
subscription.subscribe(topic='CameraGateway.{}.Frame'.format(camera_id))

# for puslish prediction images
#puslish_name = 'Images.YOLOv5'
#subscription_yolo = Subscription(channel=channel, name=puslish_name)


# to determine fps  
start_time = 0
end_time = 1

# If true recording the frames (bool)
recording = False

# If True activate the detector (bool)
detector_activated = True


gpu_activated = torch.cuda.is_available() # check if GPU is avaiable (bool)

# log info of GPu situation
if not gpu_activated:
    print('GPU is not enabled')
    device = 'cpu'
else:
    print('GPU is enabled')
    device = 'cuda:0'



# only runs when GPU is on 
while gpu_activated:

    msg = channel.consume() # consume message from channel 
    im = msg.unpack(Image) # unpack Image format (Image)
    frame = to_np(im) # Image for numpay (np.ndarray)

    
    max_size = frame.shape[1] # max size of prediction

    end_time = time.time()


    fps = int( 1 / (end_time - start_time) ) # fps of video

     # write the fps in the frames
    display_image = cv2.putText(frame, f'fps: {fps}', (5, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(250,250,250), thickness=2, lineType=cv2.LINE_AA)

    
    if detector_activated:

        detection_weapons = detector.detect_weapons(frame, max_size) # prediction weapons
        # draw bounding box in the frame
        frame = bounding_box(frame, detections=detection_weapons, class_names=detector.class_names, infer_conf=detector.weapons_detector.conf)

        detection_people = detector.detect_people(frame, max_size, recording) # prediction people

        if not recording:
            # draw bounding box in the frame
            display_image = bounding_box(frame, detections=detection_people, class_names=detector.class_names, infer_conf=detector.people_detector.conf)


    cv2.imshow('YOLO', frame) # display images
    key = cv2.waitKey(1)

    start_time = time.time()

   



    #msg2publish = Message(content=frame, reply_to=subscription_yolo)

    #channel.publish(msg2publish, topic='YoloPrediction.Frame') # Publish the frame with the yolo prediction and bounding box
    



#       xmin       ymin        xmax        ymax      confidence  class  name
#0   81.855957   1.319438  184.182846  109.907242    0.850786      0  person
#1    0.000000  23.669359   62.045010  143.517242    0.831257      0  person
#2  136.652161   6.607765  272.116699  179.696915    0.787901      0  person
#3   43.493324  87.764862   60.556637  105.465462    0.402087     41     cup




