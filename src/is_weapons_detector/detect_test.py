import cv2
import numpy as np
import torch

from is_utils import load_options, bounding_box
from weapons_detector import WeaponsDetector


op = load_options()
detector = WeaponsDetector(op.model)

img = cv2.imread('/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/src/is_weapons_detector/images/images_test/pistol_photo.jpeg')

img =cv2.resize(img, (1288, 728))

infer_size = img.shape[1]
print(img.shape)

detection_weapons = detector.detect_weapons(img)

print(detection_weapons)

img = bounding_box(img, detections=detection_weapons, class_names=detector.class_names, infer_conf=detector.weapons_detector.conf)

cv2.imshow('Image', img)
cv2.waitKey(0)

cv2.destroyAllWindows()



