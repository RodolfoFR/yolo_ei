import numpy as np
import cv2
import torch
from is_wire.core import Channel,Subscription,Message
from conf.msgs_pb2 import Image
from is_utils import to_np, save_video, random_hex_id



broker_uri = "amqp://guest:guest@localhost:5672"
channel = Channel(broker_uri) 

subscription = Subscription(channel=channel)
subscription.subscribe(topic='YoloPrediction.Frame') # subscribe  to the topic YoloPrediction.Frame for receiving predictions images

# Variables
gpu_activated = torch.cuda.is_available() # check if GPU is avaiable (bool)
video_name = random_hex_id() # name of video in hex humber (string)
path_save = '/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/videos_yolo/frames' # path to save the frames (string)
recording = False # whether record is True else False (bool)
id_frame = 0 # frame id (Int)

print(f'GPU is {gpu_activated}')

# only runs when GPU is on 
while gpu_activated: 

    msg = channel.consume()  # consume the message from the channel (Message)
    frame = msg.unpack(Image) # frame received from the channel (Image)

    frame = to_np(frame) # transformation Image for numpy, for display (np.ndarray)
                           
    cv2.imshow('Yolo frames', frame) # display frames

    key = cv2.waitKey(1) 

    # Stopping Running
    if key == ord('q'): 
        break
    
    # Save frames if key is 's' and key is 'p' pause the recording
    id_frame, recording = save_video(path_save, frame, key, video_name, id_frame, recording=recording)

        