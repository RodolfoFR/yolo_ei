import cv2
import numpy as np
import torch
import time
import json
import sys
from ImageBuffer import VideoCapture
from sort.sort_test import *
from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")   # suppress CUDA warnings

# config_file = sys.argv[1] if len(sys.argv) > 1 else '../etc/conf/config.json'
config_file = sys.argv[1] if len(sys.argv) > 1 else '/home/jvbrito/Área de Trabalho/Arquivos IC/Arquivos Porteirotron/personDetection/etc/conf/config-v5s.json'
config = json.load(open(config_file, 'r'))

# Model
model = torch.hub.load(config["model.offline"], config["model.version"],source='local', force_reload=True)     # load from local

model.conf = config["threshold"]    # confidence threshold (0-1)
model.iou = config["NMS"]           # NMS IoU threshold (0-1)
model.classes = config["classes"]   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

# Video
# cap = cv2.VideoCapture(config["video.path"])
cap = cv2.VideoCapture("./video/videocortado.mp4")
pTime = time.time()


# Comandos para salvar a saída em video

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))

# Sort object
# max_age: "Maximum number of frames to keep alive a track without associated detections.""
# min_hits: "Minimum number of associated detections before track is initialised."
# iou_threshold: "Minimum IOU for match."
people_tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3) # Mudar parametros depois. Esses são só pra ver se vai funcionar mesmo.

pts_buffer = 50
pts = [deque(maxlen=pts_buffer) for _ in range(1000)]
tam = [deque(maxlen=pts_buffer) for _ in range(1000)]

# pegando um colour map
cmap = plt.get_cmap('tab20b') # 20 cores, vai de 0 a 1.
colors = [cmap(i)[:3] for i in range(20)] # colocando esses valores em uma lista.


while True:
    success,frame = cap.read()
    if success == False:    break

    # Inference
    result = model(frame, size=640)  # includes NMS
    result = result.xyxy[0]
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime

    detection_list = [] 
    
    if len(result) != 0:
        for detection in result:
            obj = detection.cpu().numpy()
            detection_list.append(obj)
        obj = np.asarray(detection_list)
    
    # Caso nao tenha nenhuma detecção, precisamos passar um array vazio (exigência do SORT)
    else:
        obj = np.empty((0,5))

    track_bbs_ids = people_tracker.update(obj) # Atualizando o tracker com os resultados, retorna as bounding boxes e o ID único.

    # print(track_bbs_ids)
    if len(track_bbs_ids) != 0:
        for track in track_bbs_ids: # Para cada pessoa na lista de Id's observados
            #0 1
            

            #print (f'track {track}')
            x1,y1,x2,y2 = int(track[0]),int(track[1]),int(track[2]),int(track[3]) #list(map(int,track[:4]))
            id = str(track[4])
            index_id = int(float(id))
            


            color = colors[index_id % len(colors)] # Escolhe a cor baseada no index. Pega o resto da divisão inteira de index por len(colors).
            color = [i * 255 for i in color] # Passando para a escala RGB de 0 a 255.

            cv2.rectangle(frame, (x1,y1),(x2,y2), color=color, thickness=4)# Bounding Box


            if y1 > 20:  
                cv2.putText(frame, id, (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            else:        
                cv2.putText(frame, id, (x2+5,20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Printando as linhas em cada bbox.
            center:tuple = ( int((x2-x1)/2) + x1, int((y2-y1)/2) + y1)
            pts[index_id].appendleft(center)

            for i in range(1, len(pts[index_id])): #?
                if pts[index_id][i-1] is None or pts[index_id][i] is None:
                    continue
                thickness = int(np.sqrt(pts_buffer / float(i + 1)) * 2)
                cv2.line(frame, pts[index_id][i - 1], pts[index_id][i], color, thickness)
            
            #---------------------------------------------------------------------------
            # VERIFICAR SE A PESSOA ESTÁ PARADA E IMPRIMIR AVISO NA TELA
            #---------------------------------------------------------------------------
            tamvariando = 0
            distvariando = 0
            BBpequena = 0
            timeInspection = 45 #verifica se a pessoa ficou parada em 45 frames
            #Xtela = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            #Ytela = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
            Xtela = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
            Ytela = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            Dtela = np.sqrt((Xtela**2)+(Ytela**2)) 
            print(Xtela, Ytela, f'Diagonal Tela: {Dtela}')

            tamanhoBB = np.sqrt(((x2-x1)**2)+((y2-y1)**2))
            tam[index_id].appendleft(tamanhoBB)
            print (f'tamanho: {tamanhoBB/Dtela}')
            if len(tam[index_id]) <= timeInspection  or tamanhoBB is None:
                continue
            tamdif = abs((tam[index_id][0])-(tam[index_id][timeInspection]))/(tam[index_id][0])
            

            if len(pts[index_id]) <= timeInspection  or center is None:
                continue
            distancia = np.sqrt(((pts[index_id][0][0]-pts[index_id][timeInspection][0])**2) + ((pts[index_id][0][1]-pts[index_id][timeInspection][1])**2))
            # print (f'distancia: {distancia}')
            # print (f'center: {pts[index_id][0]}')
            # print (f'center_antes: {pts[index_id][timeInspection]}')
             
            #if distancia < 25:
            if distancia < (Dtela*0.034227183):
                distvariando = 1            
            if tamdif < 0.1:
                tamvariando = 1
            if tamanhoBB < (Dtela*0.1):
                BBpequena = 1
            if (tamvariando == 1) and (distvariando == 1) and (BBpequena == 0):
                cv2.putText(frame, "PARADO", (x1,y1-25), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)  #alerta impresso na tela

                


    cv2.putText(frame, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()