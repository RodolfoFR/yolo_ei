import cv2 
import sys
import os
import numpy as np
from is_utils import load_options, bounding_box
from weapons_detector import WeaponsDetector



# Check if there are any ARGS
if  not len(sys.argv)>1:
    print('Error: LACK OF ARGS')
    print('Please: When running this script put the path of your video file') 

else:
    path_video = sys.argv[1]
    # check if video exists
    if not os.path.exists(path_video):
        print('Error: Video File  does not exist')
        print('Please: Put the path of your video file')

    else:

        op = load_options() # load parameters
        detector = WeaponsDetector(op.model) 
        video = cv2.VideoCapture(path_video) 

        while video.isOpened(): # when video is opened

            ret, frame = video.read() # get frames from video

            max_size = frame.shape[1] # max size for detect

            detection_weapons = detector.detect_weapons(frame, max_size) # weapon prediction
            # draw bounding box in the frame, according prediction
            frame = bounding_box(frame, detections=detection_weapons, class_names=detector.class_names, infer_conf=detector.weapons_detector.conf) 

            detection_people = detector.detect_people(frame, max_size) # people prediction
            # draw bounding box in the frame, according prediction
            frame = bounding_box(frame, detections=detection_people, class_names=detector.class_names, infer_conf=detector.people_detector.conf)



            cv2.imshow("Yolo", frame) # display frames
            key = cv2.waitKey(1)


video.release() # close video file
cv2.destroyAllWindows() # destroy all windows cv2
