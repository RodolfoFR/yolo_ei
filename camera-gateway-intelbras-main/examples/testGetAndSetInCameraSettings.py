from is_wire.core import Channel, Subscription, Message
from is_msgs.camera_pb2 import CameraConfig, CameraConfigFields
from is_msgs.common_pb2 import FieldSelector
from google.protobuf.empty_pb2 import Empty
import json
import socket

# import test field camera 
from is_msgs.camera_pb2 import CameraSettings
from is_msgs.camera_pb2 import CameraSetting

# import test field FPS 
from is_msgs.common_pb2 import SamplingSettings
                        
                        
                        # The Bellow values ​​are default values ​​of the camera
def CameraSettingsValue(brightness_value=0.5,gamma_value=0.5,hue_value=0.5,
                        saturation_value=0.5,contrast_value=0.5,gain_value=0.02,
                        whiteBalance_value="Auto"):
                      
    brightness = CameraSetting()
    gamma = CameraSetting()
    hue = CameraSetting()
    saturation = CameraSetting()
    gain = CameraSetting()
    contrast = CameraSetting()
    whiteBalance = CameraSetting()

    brightness.ratio = brightness_value
    gamma.ratio = gamma_value
    hue.ratio = hue_value
    saturation.ratio = saturation_value
    contrast.ratio = contrast_value
    gain.ratio = gain_value
    whiteBalance.option = whiteBalance_value

    return CameraSettings(
        brightness=brightness,
        gamma=gamma,
        hue=hue,
        saturation=saturation,
        gain=gain,
        contrast=contrast,
        white_balance_bu=whiteBalance
        )

if __name__ == "__main__":
    config_path = '../etc/conf/config.json' 
    print('---RUNNING SIMPLE EXAMPLE OF RPC CONSUMPTION OF CAMERA MSGS---')
    #print('camera service should be running and broker address should be {}'.format(config['broker_uri'] ))
    print(f'client configurations on path: {config_path}')

    config = json.load(open(config_path, 'r'))
    channel = Channel(config["broker_uri"])
    subscription = Subscription(channel)

    # ---------------------- Set Config ----------------------
    fps = SamplingSettings()
    fps.frequency.value = 30
    msg_config = CameraConfig(camera = CameraSettingsValue(), 
                            sampling=fps)

    cameraId = config["camera"]["id"]
    channel.publish(Message(content=msg_config, reply_to=subscription),topic=f"CameraGateway.{cameraId}.SetConfig")

    try:
        reply = channel.consume(timeout=7.0)
        struct = reply.unpack(Empty)
        print('RPC Status:', reply.status, '\nReply:', struct)
    except socket.timeout:
        print('No reply :(')
    
        
    # ---------------------- Get Config ----------------------
    
    
    selector = FieldSelector(fields=[CameraConfigFields.Value("ALL")])
    channel.publish(
        Message(content=selector, reply_to=subscription),
        topic=f"CameraGateway.{cameraId}.GetConfig")

    try:
        reply = channel.consume(timeout=7.0)
        unpacked_msg = reply.unpack(CameraConfig)
        print(unpacked_msg)
        #print('RPC Status:', reply.status, '\nReply:', unpacked_msg)
    except socket.timeout:
        print('No reply :(')
    