import re
import sys
import cv2
import numpy as np
from datetime import datetime
# before was .conf.msgs_pb2 in from to import Image and ObjectAnnotation, original file
from conf.msgs_pb2 import Image, ObjectAnnotations
from google.protobuf.json_format import Parse
from is_wire.core import Logger, AsyncTransport
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter

from conf.options_pb2 import WeaponDetectorOptions

def annotate_image(frame, detections, class_names):
   
    h,w = frame.shape[:2] # pega a dimensão das imagens?
    image = to_image(frame) # transforma para formato png ou jpeg
    # ajusta a imagem
    image.resolution.width = w
    image.resolution.height = h
   
    # se tiver alguam detecção
    if len(detections)>0:
        obs = ObjectAnnotations()
        for det in detections:
            if len(det) == 0:continue
            (x1, y1, x2, y2,conf, clf) = det
            ob = obs.objects.add()
            v1 = ob.region.vertices.add()
            v1.x = x1
            v1.y = y1
            v2 = ob.region.vertices.add()
            v2.x = x2
            v2.y = y2
            ob.label = class_names[int(clf)]
        image.annotations.CopyFrom(obs)
       
       
    return image

def increase_bbox(img_shape,bboxes, borders = 0.0):
    if (len(bboxes)==0) or (bboxes is None): return bboxes
    bboxes = bboxes.reshape(-1,bboxes.shape[-1]).copy()
    if borders >0:
        h,w  = img_shape
        shift = (bboxes*borders).astype(int)
        bboxes[...,[0,1]] = np.where((bboxes[...,[0,1]]-shift[...,[0,1]]) > 0, bboxes[...,[0,1]]-shift[...,[0,1]],bboxes[...,[0,1]])
        bboxes[...,2] = np.where((bboxes[...,2]+shift[...,2]) <=w, bboxes[...,2]+shift[...,2],bboxes[...,2])
        bboxes[...,3] = np.where((bboxes[...,3]+shift[...,3]) <=h, bboxes[...,3]+shift[...,3],bboxes[...,3])
    return bboxes


def crop_image(image,rois):
    if rois is None: return img
    rois = rois.reshape(-1,rois.shape[-1])
    cropped_images = [image[r[1]:r[3],r[0]:r[2]]  for r in rois]
    return cropped_images
   


def get_topic_id(topic):
    values = str(topic).split(".")
    return values[1] if len(values)==3 else None

# parametros: o nome do logger (rpc server) e ip do zipkin
def create_exporter(service_name, uri):
    # cria um outro logger
    log = Logger(name="CreateExporter")
    # verica se uri esta escrito certo com http: e essas coisas
    zipkin_ok = re.match("http:\\/\\/([a-zA-Z0-9\\.]+)(:(\\d+))?", uri)
    # caso zipkin_ok = False, manda mensagem de erro
    if not zipkin_ok:
        log.critical("Invalid zipkin uri \"{}\", expected http://<hostname>:<port>", uri)
    # talvez seja um mesmangem de ok para zipkin, onde usar service_name, outras infos do zipkin
    # é uma classe, onde que passa como parametros:
    # service_name (nome do logger), hostname e porta do zipkin, transport que ele vai ter  
    exporter = ZipkinExporter(service_name=service_name,
                              host_name=zipkin_ok.group(1),
                              port=zipkin_ok.group(3),
                              transport=AsyncTransport)
    return exporter


def load_options():
    # cria um logger, rpc server
    log = Logger(name='LoadingOptions')
   
    # sys.argv é uma lista onde cada elemento é uma string, [0] é nome do arquivo python e os demais são argumento da chamada do programa
    # Se sys.arv tiver algum arguemento, então op_file é primeiro argumento da chamada
    # Se não op_file é o arquivo options.json que está em etc/conf
    # op_file é para ser um arq que passa ip das cameras(espaço), ip do zipkin_uri e os paramentros do modelo
    op_file = sys.argv[1] if len(sys.argv) > 1 else '/home/rodolfo/desenvolvimento2/espaco_inteligente/yolo_ei/etc/conf/options.json'
   
    try:
    # f = op_file, f abre op_file para leitura
        with open(op_file, 'r') as f:
            try:
            # op é uma variavel WeaponDetectorOptions(), que vem de conf/options.proto
            # nessa classe: str .broker_uri = ip da camera (ou espaço), str .zipkin_uri = ip do zipkin_uri, YoloModel .model = parametro do modelo
                op = Parse(f.read(), WeaponDetectorOptions())
                # faz o log info que passar a variavel op
                log.info('WeaponDetectorOptions: \n{}', op)
                return op
            # se o arquivo json m estiver no formato da classe WeaponDetectorOptions ele retorna mensagem de erro
            except Exception as ex:
                log.critical('Unable to load options from \'{}\'. \n{}', op_file, ex)
    except Exception as ex:
        log.critical('Unable to open file \'{}\'', op_file)

def to_np(input_image):

    """  
    Args: 
        input_image (np.ndarray or Image(is_msgs))
    
    Description:
        Tranforma a entrada em uma saída do tipo array numpy (no formato imagem, dimensionada)

    Returns:
        output_image (np.ndarray)
    """

    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


def to_image(input_image, encode_format='.jpeg', compression_level=0.8):
    if isinstance(input_image, np.ndarray):
        if encode_format == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * (100 - 0) + 0)]
        elif encode_format == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * (9 - 0) + 0)]
        else:
            return Image()
        cimage = cv2.imencode(ext=encode_format, img=input_image, params=params)
        return Image(data=cimage[1].tobytes())
    elif isinstance(input_image, Image):
        return input_image
    else:
        return Image()


def draw_detection(image, annotations):
    if annotations is None: return image
    for obj in annotations.objects:
        x1 = int(obj.region.vertices[0].x)
        y1 = int(obj.region.vertices[0].y)
        x2 = int(obj.region.vertices[1].x)
        y2 = int(obj.region.vertices[1].y)
        label = str(obj.label)
        if not (label == "Arma"):continue
        color  = (0,150,0) if label == "pessoa" else (0,0,150) if label == "Arma" else (150,0,0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
        y1 =  y1 - (text_size[1]+15) if (y1 - (text_size[1]+15)) > 0 else y1+3
        (x2, y2) = x1+text_size[0]+5, y1+text_size[1]+10
        cv2.rectangle(image, (x1+3,y1), (x2, y2), (200,200,200), -1)  # filled
        cv2.putText(image, label, (x1+3,y1+text_size[1]+4 ), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image


def generate_default_image(size = (1920,180), text = "Aguardando Conexao!!!", title = "Videomonitoramento", with_header = True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 2, 2)[0]
    shiftx, shifty= textsize[0]//2, textsize[1]//2
    w,h = size
    header = 70 if with_header else 0
    img =np.zeros((h+header,w,3), np.uint8)
    textX, textY = w//2-shiftx, h//2-shifty+header
    cv2.putText(img, text,(textX, textY), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    if with_header:
      img[5:header-15] = 255
      img[header-5:header] = 255
      plot_date(img)
      plot_logos(img, title)
    return img

def plot_date(frame):
    x1, y1 = 5, 10

   # if self.frame.ndim == 3 else cv2.merge((self.frame,self.frame,self.frame))
    date = datetime.now()
    text = "{}".format(date.strftime("%d-%m-%Y %H:%M:%S"))
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    x2,y2 = t_size[0] + 5, 50
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), -1, cv2.LINE_AA)  # filled
    cv2.putText(frame, text, (x1,y1+t_size[1]//2+t_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

def plot_logos(frame, title = "Videomonitoramento"):
    logos = [("static/logo_pmes.jpg",(50,45)), ("static/logo_viros.jpg",(70,50)), ("static/logo_ufes.png", (70,45))]
    x_before = None
    pixels_between = 30
    fw = frame.shape[1]
    x = fw
    for idx,(logo, logo_shape) in enumerate(logos):
        logo = cv2.resize(cv2.imread(logo),logo_shape)
        if frame.ndim == 2:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        lw,lh = logo_shape
        x -= (lw + pixels_between)
        shift = (70-lh)//2 -5
        frame[shift:lh+shift, x:x+lw] = logo
        t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
        x1 = (frame.shape[1]//2) - (t_size[0]//2) + 100
        y1 = (70-t_size[1])//2
        x2,y2 = t_size[0] + 5, t_size[1] + 30
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), -1, cv2.LINE_AA)  # filled
        cv2.putText(frame, title, (x1,10+t_size[1]//2+t_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
       
def prepare_to_display(img, new_shape=(640, 640), color=114, title = None, border = 6, border_color = (255,255,255)):
    h,w = img.shape[:2]
    shift_border = 4
    c = None if img.ndim ==2 else img.shape[2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / w, new_shape[1] / h)
    ratio = r, r  # width, height ratios
    new_unpad = int(round(w * r))-shift_border, int(round(h * r))-shift_border
    dw, dh = (new_shape[0] - new_unpad[0])//2, (new_shape[1] - new_unpad[1])//2  # wh padding
    if [w,h] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    nw,nh = new_unpad
    top, left = dh, dw
    bottom, right = nh+dh, nw+dw
    img_border = np.empty((new_shape[1], new_shape[0],c)) if c is not None else np.empty((new_shape[1], new_shape[0]))
    img_border.fill(color)
    img_border[top:bottom, left:right] = img
   
    if title is not None:
        x1 = y1 = shift_border + border if border is not None else shift_border
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=2)[0]
        (x2, y2) = x1+text_size[0]+5, y1+text_size[1]+10
        cv2.rectangle(img_border, (x1+3,y1), (x2, y2), (255,255,255) , -1)  # filled
        cv2.putText(img_border, title, (x1+3,y1+text_size[1]+4 ), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

    if border:
        shift = border//2+1
        cv2.rectangle(img_border,(shift,shift),(new_shape[0]-shift,new_shape[1]-shift), border_color, int(border) )
    return img_border, ratio, (dw, dh)
