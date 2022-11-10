import torch
import yolov5
import numpy as np
import cv2
from weapons_detector import WeaponsDetector
from is_utils import load_options, annotate_image


print('torch foi importado')

print(torch.cuda.is_available())



network = torch.hub.load('ultralytics/yolov5', "yolov5s",autoshape=True, pretrained=True, force_reload=True)


img = cv2.imread('images/images_test/image_people_test.jpeg')

img_np = np.asarray(img)
print(img_np)


result = network(img)

print(result.pandas().xyxy[0])

x1, y1, x2, y2, conf, classe, name = result

print(x1)
print(y1)
print(x2)
print(y2)
print(conf)
print(classe)
print(name)



#       xmin       ymin        xmax        ymax      confidence  class  name
#0   81.855957   1.319438  184.182846  109.907242    0.850786      0  person
#1    0.000000  23.669359   62.045010  143.517242    0.831257      0  person
#2  136.652161   6.607765  272.116699  179.696915    0.787901      0  person
#3   43.493324  87.764862   60.556637  105.465462    0.402087     41     cup




