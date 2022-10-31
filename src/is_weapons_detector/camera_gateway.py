                                                                                               
import sys
from time import time, sleep
from io import BytesIO
from PIL import Image as pil_image
from is_wire.core import Channel, Message, Logger
from is_msgs.image_pb2 import Image
import cv2

log = Logger(name="CameraGateway")
options = {
  # before was "amqp://10.10.2.30:30000", original file
  "broker_uri": "amqp://guest:guest@localhost:5672",
  "broker_reconnection": True,
  # before was 1, original file
  "camera_id": 1,
  "device": "/dev/video0",
  # before was [1920, 1080], original file
  "resolution": [1920, 728],
  "fps": 10,
  "flip_h": True,
  "flip_v": False
}


if len(sys.argv)>0:
    options['camera_id'] = sys.argv[1]
if len(sys.argv)>2:
    video_path = sys.argv[2]

params = [cv2.IMWRITE_JPEG_QUALITY, 90]

def run():
    channel = Channel(options['broker_uri'])
    log.info("Connected to {}", options['broker_uri'])
    camera = cv2.VideoCapture(video_path)
    while True: 
        t0 = time()
        grab, img = camera.read()
        if not grab:break
        img = cv2.resize(img, options['resolution'])
        ret,img = cv2.imencode(ext=".jpeg", img=img, params=params)
        pb_image = Image(data=img.tobytes()  )
        msg = Message()
        msg.pack(pb_image)
        channel.publish(msg, topic="CameraGateway.{}.Frame".format(options['camera_id']))
        dt = time() - t0
        log.debug("took_ms={:.1f}", dt * 1e3)
        interval = max(0.0, 1.0 / options['fps'])
        sleep(interval)


run()

