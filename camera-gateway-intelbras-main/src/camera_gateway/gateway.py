from is_wire.core import Channel, Message, Logger,StatusCode,AsyncTransport,Tracer
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.rpc import ServiceProvider, LogInterceptor
from is_msgs.camera_pb2 import CameraConfig, CameraConfigFields
from is_msgs.common_pb2 import FieldSelector
from google.protobuf.empty_pb2 import Empty

import socket

def get_obj(callable, obj):
    value = callable()
    if value is not None:
        obj.CopyFrom(value)


def get_val(callable, obj, attr):
    value = callable()
    if value is not None:
        setattr(obj, attr, value)

class CameraGateway(object):
    def __init__(self, driver):
        self.driver = driver
        self.logger = Logger("CameraGateway")
    
    def get_config(self, field_selector, ctx):
        fields = field_selector.fields
        camera_config = CameraConfig()

        if CameraConfigFields.Value("ALL") in fields or \
           CameraConfigFields.Value("STREAM_CHANNEL_ID") in fields:
            get_val(self.driver.get_stream_channel_id,
                    camera_config.stream_channel_id, "value")

        if CameraConfigFields.Value("ALL") in fields or \
           CameraConfigFields.Value("SAMPLING_SETTINGS") in fields:
            get_val(self.driver.get_fps,
                    camera_config.sampling.frequency, "value")

        if CameraConfigFields.Value("ALL") in fields or \
           CameraConfigFields.Value("IMAGE_SETTINGS") in fields:
            get_obj(self.driver.get_resolution, camera_config.image.resolution)

        if CameraConfigFields.Value("ALL") in fields or \
           CameraConfigFields.Value("CAMERA_SETTINGS") in fields:
            get_val(self.driver.get_brightness, camera_config.camera.brightness, "ratio")
            get_val(self.driver.get_contrast, camera_config.camera.contrast, "ratio")
            get_val(self.driver.get_gamma, camera_config.camera.gamma, "ratio")
            get_val(self.driver.get_hue, camera_config.camera.hue, "ratio")
            get_val(self.driver.get_saturation, camera_config.camera.saturation, "ratio")
            get_val(self.driver.get_gain, camera_config.camera.gain, "ratio")
            get_val(self.driver.get_whiteBalance, camera_config.camera.white_balance_bu, "option")

        return camera_config

    def set_config(self, camera_config, ctx):

        if camera_config.HasField("stream_channel_id"):
            return self.driver.set_stream_channel_id(camera_config.stream_channel_id.value)
            
        if camera_config.HasField("sampling"):
            if camera_config.sampling.HasField("frequency"):
                maybe_ok = self.driver.set_fps(camera_config.sampling.frequency.value)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok
            
        if camera_config.HasField("image"):
            if camera_config.image.HasField("resolution"):
                maybe_ok = self.driver.set_resolution(camera_config.image.resolution)
                if maybe_ok != StatusCode.OK:
                    return maybe_ok

        if camera_config.HasField("camera"):
            if camera_config.camera.HasField("brightness"):
                maybe_ok = self.driver.set_brightness(camera_config.camera.brightness.ratio)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok

            if camera_config.camera.HasField("gamma"):
                maybe_ok = self.driver.set_gamma(camera_config.camera.gamma.ratio)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok
            
            if camera_config.camera.HasField("hue"):
                maybe_ok = self.driver.set_hue(camera_config.camera.hue.ratio)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok

            if camera_config.camera.HasField("saturation"):
                maybe_ok = self.driver.set_saturation(camera_config.camera.saturation.ratio)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok

            if camera_config.camera.HasField("gain"):
                maybe_ok = self.driver.set_gain(camera_config.camera.gain.ratio)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok

            if camera_config.camera.HasField("contrast"):
                maybe_ok = self.driver.set_contrast(camera_config.camera.contrast.ratio)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok

            if camera_config.camera.HasField("white_balance_bu"):
                maybe_ok = self.driver.set_whiteBalance(camera_config.camera.white_balance_bu.option)
                if maybe_ok.code != StatusCode.OK:
                    return maybe_ok           

        return Empty()


    def run(self, broker_uri):
        service_name = "CameraGateway.{}".format(self.driver.camera_id)
        
        publish_channel = Channel(broker_uri)
        
        exporter = ZipkinExporter(
            service_name=service_name,
            host_name=self.driver.zipkin_url,
            port=self.driver.zipkin_port,
            transport=AsyncTransport,
            )
        rpc_channel = Channel(broker_uri)
        server = ServiceProvider(rpc_channel)

        logging = LogInterceptor()
        server.add_interceptor(logging)

        server.delegate(
            topic=service_name + ".GetConfig",
            request_type=FieldSelector,
            reply_type=CameraConfig,
            function=self.get_config)

        server.delegate(
            topic=service_name + ".SetConfig",
            request_type=CameraConfig,
            reply_type=Empty,
            function=self.set_config)

        self.logger.info("RPC listening for requests")

        while True:
            tracer = Tracer(exporter)
            message = Message()
            with tracer.span(name="frame") as span:
                image = self.driver.grab_image()
                msg = Message(content=image)
                msg.inject_tracing(span)
            if len(image.data) > 0:
                publish_channel.publish(msg, topic=service_name + ".Frame")
            else:
                self.logger.warn("No image captured.")
            try:
                message = rpc_channel.consume(timeout=0)
                if server.should_serve(message):
                    server.serve(message)
            except socket.timeout:
                pass
            
                
