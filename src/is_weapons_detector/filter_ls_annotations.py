

import json
import numpy as np 
from glob import glob
import random
import shutil
import cv2




# labels = glob("/public/datasets/videomonitoramento-ifes/annotations/yolo_format/*.txt")
# size = len(labels)
# indices = list(range(size))
# random.shuffle(indices)
# test_indices = indices[:int(0.2*size)]
# train_indices = indices[int(0.2*size):]
# image_src  = "/public/datasets/videomonitoramento-ifes/armas/"
# image_dest = "images/train"
# label_dest = "labels/train"
# for idx in train_indices:
#     tf = labels[idx]
#     label = tf.split("/")[-1]
#     image_name = f'{label.split(".")[0]}.png'
#     shutil.copyfile(f"{image_src}/{image_name}", f"{image_dest}/{image_name}")
#     shutil.copyfile(tf, f"{label_dest}/{label}")

# image_dest = "images/test"
# label_dest = "labels/test"
# for idx in test_indices:
#     tf = labels[idx]
#     label = tf.split("/")[-1]
#     image_name = f'{label.split(".")[0]}.png'
#     shutil.copyfile(f"{image_src}/{image_name}", f"{image_dest}/{image_name}")
#     shutil.copyfile(tf, f"{label_dest}/{label}")




    


    


annotations = glob("./annotations/8990*.json")
email_filter = None #["clebeson.canuto@gmail.com"]
# filter_labels = ["car",",motocycle", "truck","bus","bicycle"]
filter_labels =  ["Pistol","Armed-person"]
total = 0
with_vehicles = 0
total_vehicles = 0
entry_labels = []
image = cv2.imread("./annotations/cam2_v1_21.png")
height,width,c = image.shape
print(height,width,c)
for file in annotations:
    with open(file,"r") as f:
        annotation = json.load(f)
    results = annotation["result"]
    image_name = annotation["task"]["data"]["image"].split("/")[-1]
    entity = [image_name]
    has_choice_label = False
    
    if email_filter is None or annotation["completed_by"]["email"] in email_filter:
        total += 1
        num_vehicles = 0
        entry_label = []
        for result in results:
            if result["type"] == "choices":
                has_choice_label = True
            elif result["type"] == "rectanglelabels": #and result["from_name"] == "type-vel":
                value = result["value"]
                label = value.get("rectanglelabels",[])
                label = "" if len(label) == 0 else label[0]
                if label in filter_labels:
                    w = value["width"]/100.0
                    h = value["height"]/100.0
                    x = ((value["x"]/100.0) + w/2)
                    y = ((value["y"]/100.0) + h/2)
                    # x1 = int(width * (x - w / 2))  # top left x
                    # y1 = int(height * (y - h / 2))   # top left y
                    # x2 = int(width * (x + w / 2))  # bottom right x
                    # y2 = int(height * (y + h / 2))   # bottom right y
                    
                    # cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 1)
                    # cv2.imshow("image", image)
                    # cv2.waitKey()
                    entry_label.append(f"{filter_labels.index(label)} {x} {y} {w} {h}")
                    num_vehicles += 1

        if not has_choice_label:
            total_vehicles += num_vehicles
            with_vehicles += 0 if num_vehicles == 0 else 1
            # for entry in entry_label:
            #     print(entry)
                
            with open(f'./annotations/yolo_format/{image_name.split(".")[0]}.txt', 'a') as the_file:
                for entry in entry_label:
                    the_file.write(f"{entry}\n")
print(f"Total labeled images = {total}    images with vehicles {with_vehicles}  skipped images {total-with_vehicles} total vehicles {total_vehicles}")

            
        




