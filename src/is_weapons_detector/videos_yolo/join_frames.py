import os
import sys
import cv2

# variables
frames = [] # list of frames
n_files = 0 # number of files
height = 0 
width = 0

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
        for root, dirs, files in os.walk(folder, topdown=True):
            files.sort()
            n_files = len(files)

        
        pattern_file_neme = files[0][0:16] 

        for i in range(n_files):

            # save the frame in correct order
            path = os.path.join(root,f'{pattern_file_neme}{i}.png') 
            frame = cv2.imread(path)
            frames.append(frame)
            height += frame.shape[0]
            width += frame.shape[1]
       
        # average file height and width         
        height = int(height/n_files)
        width = int(width/n_files)

        # FPS that the user wants
        fps = int(input('What is the FPS of the video you want:    '))

        # crete variable video
        video = cv2.VideoWriter('the_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

        # Put the save frame in the video
        for i in range(0, n_files):
            video.write(frames[i])



