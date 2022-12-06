import os
import sys
import cv2

# variables
frames = [] # list of frames
n_files = 0 # number of files
height = 0 
width = 0
arvs_frame = []
files = []

# Check if there are any ARGS
if  not len(sys.argv)>1:
    print('Error: LACK OF ARGS')
    print('Please: When running this script put the path of your file folder') 

else:
    folder = sys.argv[1]
    # check if folder exists
    if not os.path.exists(folder):
        print('Error: File folder does not exist')
        print('Please: Put the path of your file folder')

    else:



        # save the name of the files of folder and your path
        for root, dirs, file in os.walk(folder, topdown=True):
 
            if len(file) > 0:
                n_files += 1

        for i in range(1, n_files+1):

            if i == 1:

                frame = cv2.imread(f'/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/videos_yolo/frames/image0.jpg')
            else:
                frame = cv2.imread(f'/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/videos_yolo/frames{i}/image0.jpg')

            frames.append(frame) # list of frames
            height += frame.shape[0]
            width += frame.shape[1]


        
        # average file height and width         
        height = int(height/len(frames))
        width = int(width/len(frames))

        # FPS that the user wants
        fps = int(input('What is the FPS of the video you want:    '))

        # crete variable video
        video = cv2.VideoWriter('the_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

        # Put the save frame in the video
        for i in range(len(frames)):
            video.write(frames[i])
