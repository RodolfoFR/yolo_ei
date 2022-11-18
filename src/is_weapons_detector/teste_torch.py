#import torch
#import yolov5
import numpy as np
import cv2
#from weapons_detector import WeaponsDetector
from random import choice


def random_hex_id():

    hex_digit = 'FEDCBA9876543210'
    hex_number= ''
    for i in range(0, 6):
        hex_number += choice(hex_digit)
        
    return hex_number

v = random_hex_id()

print(v)



#       xmin       ymin        xmax        ymax      confidence  class  name
#0   81.855957   1.319438  184.182846  109.907242    0.850786      0  person
#1    0.000000  23.669359   62.045010  143.517242    0.831257      0  person
#2  136.652161   6.607765  272.116699  179.696915    0.787901      0  person
#3   43.493324  87.764862   60.556637  105.465462    0.402087     41     cup




