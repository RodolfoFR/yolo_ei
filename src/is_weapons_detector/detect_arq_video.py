import cv2 
import sys
import os
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

        op = load_options()
        detector = WeaponsDetector(op.model)
        video = cv2.VideoCapture(path_video)

        