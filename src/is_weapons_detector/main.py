import os
import sys
import cv2
import numpy as np

import dateutil.parser as dp

# before was .conf.msgs_pb2 in from to import Image, original file
#from conf.msgs_pb2 import Image
from is_msgs.image_pb2 import Image
from is_wire.core import Logger, Subscription, Message, Tracer

from weapons_detector import WeaponsDetector
from is_utils import load_options, create_exporter, get_topic_id,  to_image, to_np, annotate_image
from stream_channel import StreamChannel





def span_duration_ms(span):
    dt = dp.parse(span.end_time) - dp.parse(span.start_time)
    return dt.total_seconds() * 1000.0

# congifurar os tamanhos tas imagens, para plotar elas juntas
def place_images(output_image, images):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0:h, 0:w, :] = images[0]
    output_image[0:h, w:2 * w, :] = images[1]
    output_image[h:2 * h, 0:w, :] = images[2]
    output_image[h:2 * h, w:2 * w, :] = images[3]


   
def main():

    # parte do rpc server, criar uma variavel logger
    service_name = "WeaponsDetector.Detect"
    log = Logger(name=service_name)
   
    # load_options(), uma funcao,  vem de is_utils.py
    # vai fazer um log.info do que vai ter dentro de op
    # basicamente op vai ser da classe WeaponDetectorOptions que vem de conf/options.proto
    # nessa classe: str .broker_uri = ip da camera (ou espaço), str .zipkin_uri = ip do zipkin_uri, YoloModel .model = parametro do modelo
    op = load_options()
   
    # detector vai ser da classe WeaponsDetector que vem de weapons_detector.py, Rede para classificar
    # que tem como parametro op.model, que sao os parametro do modelo (vai ter os pesos)
    #detector = WeaponsDetector(op.model)

    # StreamChannel é uma classe que vem de stream_channel.py
    # é parecido com Channel do is_wire.core
    # Mas ele é um canal, consume mensagem, se ele esperar demais, ele para, ou quando para a conexão
    channel = StreamChannel(op.broker_uri)
   
    # log.info do ip das cameras que ele ta conetado
    log.info('Connected to broker {}', op.broker_uri)

    # create_exporter, é uma função de ia_utils.py, parametros: o nome do logger (rpc server) e ip do zipkin
    # exporter é alguma coisa que envolver o zipkin
    #
    exporter = create_exporter(service_name=service_name, uri=op.zipkin_uri)
   
    # Subscription para o cannal, e logger service_name
    # Se escreve para o topico 'CameraGateway.*.Frame'
    subscription = Subscription(channel=channel, name=service_name)
    # subscription no topico do ip das 4 cameras
    for c in op.cameras:
        subscription.subscribe('CameraGateway.{}.Frame'.format(c.id))

    
    size = (2 * op.cameras[0].config.image.resolution.height,
            2 * op.cameras[0].config.image.resolution.width, 
            3)
    
    full_image = np.zeros(size, dtype=np.uint8)


    images_data = {}
    n_sample = 0
    display_rate = 4
    first = 0
    rod = 0
    
    while True:
   
    # parametro return_dropped=True serve além de receber a messagem receber o dropped
    # consume a mensagem  
        msg, dropped = channel.consume_last(return_dropped=True)
       
        # não sei o msg.extract_tracing(),
        tracer = Tracer(exporter, span_context=msg.extract_tracing())
        span = tracer.start_span(name='detection_and_render')
        detection_span = None

        with tracer.span(name='unpack'):
            # trata a mensagem recebido como imagem
            im = msg.unpack(Image)
            # transforma a imagem em numpy
            #im_np = to_np(im)
            im_np = np.fromstring(im.data, dtype=np.uint8)
            


        with tracer.span(name='detection') as _span:
            # pega o topic da mensagem e dividi ela
            # retorna só o numero do id (0, 1, 2, 3), se estiver correta a formatação
            camera_id = get_topic_id(msg.topic)

            # amazena as imagens(formato np) em suas posições
            images_data[camera_id] = im_np
            # classificar a imagem (formato numpy)

            
            
            #weapons = detector.detect(im_np) # retorna a coordenada da pessoa com arama, ou a propria arma,
            weapons = []
            detection_span = _span

            if len(images_data) == len(op.cameras):
                
                for c in op.cameras:
                    n_sample += 1
                
                if n_sample % display_rate == 0:

                    images = [0, 0, 0, 0]

                    for _, d in images_data.items():
                        i = int(_)
                        images[i] = cv2.imdecode(d, cv2.IMREAD_COLOR)
                   
                    place_images(full_image, images) # reajusta o full_image para receber as imagens
                    display_image = cv2.resize(full_image, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow('', display_image)

                    key = cv2.waitKey(1)

                    if key == ord('q'):
                        break

        #with tracer.span(name='image_and_annotation_publish'):
            # é a anotação da imagem classifica, de aonde está o objeto classificad, a precisão e a classe dele
                #image = annotate_image(im_np,weapons, detector.class_names)
             # publica a anotação da imagem
                #ann_image_msg = Message()
        #     ann_image_msg.topic = 'WeaponsDetector.{}.Frame'.format(camera_id)
        #     #ann_image_msg.pack(image)
        #     channel.publish(ann_image_msg)
            

        # span.add_attribute('Detections', len(weapons))
        # tracer.end_span()

        # info = {
        #     'detections': weapons,
        #     'dropped_messages': dropped,
        #     'took_ms': {
        #         'detection': round(span_duration_ms(detection_span), 2),
        #         'service': round(span_duration_ms(span), 2)
        #     }
        # }
        # log.info('{}', str(info).replace("'", '"'))


if __name__ == "__main__":
    main()
