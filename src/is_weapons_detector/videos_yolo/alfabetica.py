import os
import sys
import cv2

frames = []
n_files = 0
height = 0
width = 0

def in_order(datas):

    only_id = []
    correct_order = []
    name = datas[0][0:28]
    file_name = ''

    datas.sort()

    for data in datas:
        data = data[28:]

        for caractere in data:
            data = ''
            if caractere.isdigit():
                data += caractere

        only_id.append(data)

    only_id = sorted(only_id)

    

    for i in range(0, len(only_id)):

        file_check = name + only_id[i] + '.png'

        while len (correct_order) < len(datas[0:i+1]):
            
            for data in datas:

                if data == file_check:
                    correct_order.append(file_check)
    
    return correct_order
                    
        




if  not len(sys.argv)>1:
    print('Error: LACK OF ARGS')
    print('Please: When running this script put the path of your file folder') 

else:
    folder = sys.argv[1]
    if not os.path.exists(folder):
        print('Error: File does not exist')
        print('Please: Put the path of your file folder')

    else:

        for root, dirs, files in os.walk(folder, topdown=True):

            print(root)
            print(files)

            files = in_order(files)
            print('================================================')
            print(files)

