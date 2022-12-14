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
flag = [0, 1]

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


        if flag[0] == 1:
            # save the name of the files of folder and your path
            for root, dirs, file in os.walk(folder, topdown=True):
    
                if len(file) > 0:
                    n_files += 1

            for i in range(3, n_files+1, 4):

                if i == 1:

                    frame = cv2.imread(f'/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/videos_yolo/frames/image0.jpg')
                else:
                    frame = cv2.imread(f'/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/videos_yolo/frames{i}/image0.jpg')

                frames.append(frame) # list of frames
                height += frame.shape[0]
                width += frame.shape[1]
            video_name = 'The_video'

        elif flag[1] == 1: # second method

            for root, dirs, file in os.walk(folder, topdown=True): # count the amount of frames
                if len(file) > 0:
                    n_files = len(file)
                    ex_file = file[0]

            for i in range(n_files):

                if i == 0:
                    video_name = ex_file[0:6]
                    print(f'Video name: {video_name}')
                frame = cv2.imread(f'/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/Videos/frames/{video_name}-{i}.jpg')
                frames.append(frame) # list of frames
                height += frame.shape[0]
                width += frame.shape[1]
        
        # average file height and width         
        height = int(height/len(frames))
        width = int(width/len(frames))

        # FPS that the user wants
        fps = int(input('What is the FPS of the video you want:    '))

        # crete variable video
        video = cv2.VideoWriter(f'{video_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

        # Put the save frame in the video
        for i in range(len(frames)):
            video.write(frames[i])
