from is_wire.core import Channel, Subscription, Message
from is_msgs.camera_pb2 import CameraConfig, CameraConfigFields
from is_msgs.common_pb2 import FieldSelector
from google.protobuf.empty_pb2 import Empty
import json
import socket



if __name__ == "__main__":
    config_path = '../etc/conf/config.json' 
    print('---RUNNING SIMPLE EXAMPLE OF RPC CONSUMPTION OF CAMERA MSGS---')
    print('camera service should be running and broker address should be {}'.format(config['broker_uri'] ))
    print(f'client configurations on path: {config_path}')

    config = json.load(open(config_path, 'r'))
    channel = Channel(config["broker_uri"])
    subscription = Subscription(channel)

    # ---------------------- Set Config ----------------------

    msg_config = CameraConfig()
    msg_config.stream_channel_id.value = 1
    channel.publish(Message(content=msg_config, reply_to=subscription),topic="CameraGateway.4.SetConfig")

    try:
        reply = channel.consume(timeout=3.0)
        struct = reply.unpack(Empty)
        print('RPC Status:', reply.status, '\nReply:', struct)
    except socket.timeout:
        print('No reply :(')
        
    # ---------------------- Get Config ----------------------

    selector = FieldSelector(fields=[CameraConfigFields.Value("ALL")])
    channel.publish(
        Message(content=selector, reply_to=subscription),
        topic="CameraGateway.4.GetConfig")

    try:
        reply = channel.consume(timeout=3.0)
        unpacked_msg = reply.unpack(CameraConfig)
        print('RPC Status:', reply.status, '\nReply:', unpacked_msg)
    except socket.timeout:
        print('No reply :(')
