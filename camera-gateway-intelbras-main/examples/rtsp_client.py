from is_wire.core import Logger,Status ,StatusCode
from is_msgs.image_pb2 import Image, Resolution, ImageFormat, ImageFormats
import cv2
import av
import queue
import threading

import requests
from requests.auth import HTTPDigestAuth
from is_msgs.common_pb2 import SamplingSettings
from enum import Enum

class VideoColor(Enum):
    brightness = 1
    contrast = 2
    gamma = 3
    hue = 4
    saturation = 5

def assert_type(instance, _type, name):
    if not isinstance(instance, _type):
        raise TypeError("Object {} must be of type {}".format(
            name, _type.DESCRIPTOR.full_name))

class IntelBDriver():
    logger = Logger("IntelBDriver")
    def __init__(self, configurations):
        self.ip = configurations['ip']
        self.rtsp_port = configurations['rtsp_port']
        self.user = configurations['user']
        self.password = configurations['password']
        self.channel_id = configurations['channel']
        self.stream_channel = configurations['stream_channel']
        self.camera_id = configurations['id']
        self.http_port = configurations['http_port']

        image_format = ImageFormat()
        image_format.format = ImageFormats.Value("JPEG")
        image_format.compression.value = 0.8
        self.set_image_format(image_format)
        status_config_stream = self.__configure_stream()
        if status_config_stream != Status(StatusCode.OK):
            self.logger.critical("ERROR on camera stream initialization.")

    def __configure_stream(self):
        self.rtsp_url = "rtsp://{}:{}@{}:{}/cam/realmonitor?channel={}&subtype={}".format(self.user,self.password,self.ip,self.rtsp_port,self.channel_id,self.stream_channel)       
        retry = 1
        max_retry = 5
        while retry <= max_retry:
            self.logger.info("Connecting to camera {} ({}) with stream {}".format(self.camera_id,self.ip,self.stream_channel))
            try:
                options = {'rtsp_transport': 'tcp'}
                self.video = av.open(self.rtsp_url, 'r', options=options)
            except:
                self.logger.critical("Unable to open video stream")
            if self.video is not None:
                break
            else: 
                self.logger.error("Not conntected to camera {} ({}). Retrying {}/{}...".format(self.camera_id,self.ip,retry,max_retry))
                retry += 1
        else:
            self.logger.error("Max number of connection retries have been reached.")
            return Status(StatusCode.DEADLINE_EXCEEDED,why="Max number of connection retries have been reached.")
        return Status(StatusCode.OK) 
        

    def get_np_image(self):
        for packet in self.video.demux():
                for frame in packet.decode():
                    if packet.stream.type == 'video':
                        dur = packet.stream.base_rate
                        self.logger.info(f"duration: {dur}")
                        return frame.to_ndarray(format='bgr24')
                        

    def grab_image(self):
        frame = self.get_np_image()
        image = cv2.imencode(ext=self.encode_format,
                             img=frame, params=self.encode_parameters)
        return Image(data=image[1].tobytes())

    def set_image_format(self, image_format):
        assert_type(image_format, ImageFormat, "image_format")
        if image_format.format == ImageFormats.Value("JPEG"):
            self.encode_format = ".jpeg"
        elif image_format.format == ImageFormats.Value("PNG"):
            self.encode_format = ".png"
        elif image_format.format == ImageFormats.Value("WebP"):
            self.encode_format = ".webp"

        if image_format.HasField("compression"):
            if self.encode_format == '.jpeg':
                self.encode_parameters = [
                    cv2.IMWRITE_JPEG_QUALITY,
                    int(image_format.compression.value * (100 - 0) + 0)
                ]
            elif self.encode_format == '.png':
                self.encode_parameters = [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    int(image_format.compression.value * (9 - 0) + 0)
                ]
            elif self.encode_format == '.webp':
                self.encode_parameters = [
                    cv2.IMWRITE_WEBP_QUALITY,
                    int(image_format.compression.value * (100 - 1) + 1)
                ]

    def get_image_format(self):
        image_format = ImageFormat()
        if self.encode_format == ".jpeg":
            image_format.format = ImageFormats.Value("JPEG")
            image_format.compression.value = self.encode_parameters[1] / 100.0
        elif self.encode_format == ".png":
            image_format.format = ImageFormats.Value("PNG")
            image_format.compression.value = self.encode_parameters[1] / 9.0
        elif self.encode_format == ".webp":
            image_format.format = ImageFormats.Value("WebP")
            image_format.compression.value = (
                self.encode_parameters[1] - 1) / 99.0
        return image_format

    def set_fps(self, fps, recordType=0):
        supported_range = range(1, 31) # fps support value is between 1 to 30
        supported_range_firstValue = supported_range[0]
        if (fps in supported_range):
            fps_url = f"http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&Encode[{self.stream_channel}].MainFormat[{recordType}].Video.FPS={fps}"
            fps_request = requests.get(fps_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported fps! Received: {fps} | fps value is between {supported_range_firstValue} to {len(supported_range)}") 

    def get_fps(self):
        FPS = SamplingSettings()
        fps_url = f"http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=getConfig&name=Encode"
        fps_request = requests.get(fps_url, auth=HTTPDigestAuth(self.user, self.password)).content.decode("utf-8")
        fpsField_index = 9
        fps_value = fps_request.split()[fpsField_index][-2:] if fps_request.split()[fpsField_index][-2] != "=" else fps_request.split()[fpsField_index][-1] # get fps number value in all the cases
        FPS.frequency.value = int(fps_value)        
        return FPS.frequency.value

    def _get_videoColor(self, param):
        Weight = 100 # Allowed values in CameraSetting.ratio is between 0 to 1 and not 1 to 100, this is the why to the weight variable
        videoColor_url = f"http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=getConfig&name=VideoColor"
        videoColor_request = requests.get(videoColor_url, auth=HTTPDigestAuth(self.user, self.password)).content.decode("utf-8")

        # All the Values ​​​​below Will be Between 0 to 100
        if param == VideoColor.brightness:
            brightnessField_index = 0

            if videoColor_request.split()[brightnessField_index][-3] == '=': # value Between 10 to 99
                return int(videoColor_request.split()[brightnessField_index][-2:])/Weight

            elif videoColor_request.split()[brightnessField_index][-4] == '=': # value == 100
                return int(videoColor_request.split()[brightnessField_index][-3:])/Weight

            else: # value < 10
                return int(videoColor_request.split()[brightnessField_index][-1:])/Weight
        
        # All the Values ​​​​below Will be Between 0 to 100
        elif param == VideoColor.contrast:
            contrastField_index = 2

            if videoColor_request.split()[contrastField_index][-3] == '=': # value Between 10 to 99
                return int(videoColor_request.split()[contrastField_index][-2:])/Weight

            elif videoColor_request.split()[contrastField_index][-4] == '=': # value == 100
                return int(videoColor_request.split()[contrastField_index][-3:])/Weight

            else: # value < 10
                return int(videoColor_request.split()[contrastField_index][-1:])/Weight

        # All the Values ​​​​below Will be Between 0 to 100    
        elif param == VideoColor.gamma:
            gammaField_index = 3

            if videoColor_request.split()[gammaField_index][-3] == '=': # value Between 10 to 99
                return int(videoColor_request.split()[gammaField_index][-2:])/Weight

            elif videoColor_request.split()[gammaField_index][-4] == '=': # value == 100
                return int(videoColor_request.split()[gammaField_index][-3:])/Weight

            else: # value < 10
                return int(videoColor_request.split()[gammaField_index][-1:])/Weight
        
        # All the Values ​​​​below Will be Between 0 to 100
        elif param == VideoColor.hue:
            hueField_index = 4

            if videoColor_request.split()[hueField_index][-3] == '=': # value Between 10 to 99
                return int(videoColor_request.split()[hueField_index][-2:])/Weight

            elif videoColor_request.split()[hueField_index][-4] == '=': # value == 100
                return int(videoColor_request.split()[hueField_index][-3:])/Weight

            else: # value < 10
                return int(videoColor_request.split()[hueField_index][-1:])/Weight

        elif param == VideoColor.saturation:
            # All the Values ​​​​below Will be Between 0 to 100
            saturationField_index = 5

            if videoColor_request.split()[saturationField_index][-3] == '=': # value Between 10 to 99
                return int(videoColor_request.split()[saturationField_index][-2:])/Weight

            elif videoColor_request.split()[saturationField_index][-4] == '=': # value == 100
                return int(videoColor_request.split()[saturationField_index][-3:])/Weight

            else: # value < 10
                return int(videoColor_request.split()[saturationField_index][-1:])/Weight
    
    def get_brightness(self):
        return self._get_videoColor(VideoColor.brightness)
    
    def get_contrast(self):
        return self._get_videoColor(VideoColor.contrast)

    def get_gamma(self):
        return self._get_videoColor(VideoColor.gamma)

    def get_hue(self):
        return self._get_videoColor(VideoColor.hue)

    def get_saturation(self):
        return self._get_videoColor(VideoColor.saturation)

    def get_gain(self):
        Weight = 100 # Allowed values in CameraSetting.ratio is between 0 to 1 and not 1 to 100, this is the why to the weight variable
        gain_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=getConfig&name=VideoInOptions'
        gain_request = requests.get(gain_url, auth=HTTPDigestAuth(self.user, self.password)).content.decode("utf-8")
        gainField_index = 18
        if gain_request.split()[gainField_index][-3] == '=': # value Between 10 to 99
            return int(gain_request.split()[gainField_index][-2:])/Weight

        elif gain_request.split()[gainField_index][-4] == '=': # value == 100
            return int(gain_request.split()[gainField_index][-3:])/Weight

        else: # value < 10
            return int(gain_request.split()[gainField_index][-1:])/Weight

    def set_gain(self, gain_value):
        Weight, firstValue, lastValue = 100, 0, 100
        gain = gain_value*Weight # "gain_value" is between 0 to 1 and the allow range in API is between 1 to 100
        gain_ok = firstValue <= gain <= lastValue
        if (gain_ok):
            gain_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoInOptions[{self.stream_channel}].Gain={gain}'
            gain_request = requests.get(gain_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            firstValue_normalized = firstValue/Weight
            lastValue_normalized = lastValue/Weight           
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported gain! Received: {gain_value} | gain value is between {firstValue_normalized} to {lastValue_normalized}")


    def set_brightness(self, brightness_value):
        Weight, firstValue, lastValue = 100, 0, 100
        brightness = brightness_value*Weight # "brightness_value" is between 0 to 1 and the allow range in API is between 1 to 100
        brightness_ok = firstValue <= brightness <= lastValue
        if (brightness_ok):
            Brightness_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoColor[{self.stream_channel}][{self.stream_channel}].Brightness={brightness}'
            Brightness_request = requests.get(Brightness_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            firstValue_normalized = firstValue/Weight
            lastValue_normalized = lastValue/Weight
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported brightness value! Received: {brightness_value} | brightness value is between {firstValue_normalized} to {lastValue_normalized}")
                   

    def set_contrast(self, contrast_value):
        Weight, firstValue, lastValue = 100, 0, 100
        contrast = contrast_value*Weight # "gamma_value" is between 0 to 1 and the allow range in API is between 1 to 100
        contrast_ok = firstValue <= contrast <= lastValue
        if (contrast_ok):  
            contrast_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoColor[{self.stream_channel}][{self.stream_channel}].Contrast={contrast}'
            contrast_request = requests.get(contrast_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            firstValue_normalized = firstValue/Weight
            lastValue_normalized = lastValue/Weight 
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported contrast value! Received: {contrast_value} | contrast value is between {firstValue_normalized} to {lastValue_normalized}")           

    def set_gamma(self, gamma_value):
        Weight, firstValue, lastValue = 100, 0, 100
        gamma = gamma_value*Weight # "gamma_value" is between 0 to 1 and the allow range in API is between 1 to 100
        gamma_ok = firstValue <= gamma <= lastValue
        if (gamma_ok):
            gamma_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoColor[{self.stream_channel}][{self.stream_channel}].Gamma={gamma}'
            gamma_request = requests.get(gamma_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            firstValue_normalized = firstValue/Weight
            lastValue_normalized = lastValue/Weight           
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported gamma value! Received: {gamma_value} | gamma value is between {firstValue_normalized} to {lastValue_normalized}")            


    def set_hue(self, hue_value):
        Weight, firstValue, lastValue = 100, 0, 100
        hue = hue_value*Weight # "hue_value" is between 0 to 1 and the allow range in API is between 1 to 100
        hue_ok = firstValue <= hue <= lastValue
        if (hue_ok):
            hue_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoColor[{self.stream_channel}][{self.stream_channel}].Hue={hue}'
            hue_request = requests.get(hue_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            firstValue_normalized = firstValue/Weight
            lastValue_normalized = lastValue/Weight
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported hue value! Received: {hue_value} | hue value is between {firstValue_normalized} to {lastValue_normalized}")           

            
    def set_saturation(self, saturation_value):
        Weight, firstValue, lastValue = 100, 0, 100
        saturation = saturation_value*Weight # "saturation_value" is between 0 to 1 and the allow range in API is between 1 to 100
        saturation_ok = firstValue <= saturation <= lastValue
        if (saturation_ok):
            saturation_url = f'http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoColor[{self.stream_channel}][{self.stream_channel}].Saturation={saturation}'
            saturation_request = requests.get(saturation_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            firstValue_normalized = firstValue/Weight
            lastValue_normalized = lastValue/Weight
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported saturation value! Received: {saturation_value} | support value is between {firstValue_normalized} to {lastValue_normalized}")

    def set_whiteBalance(self, whiteBalance, ChannelNo=0):
        whiteBalance_values = {"Disable", "Auto", "Custom", "Sunny", "Cloudy", "Home", "Office", "Night"}
        if (whiteBalance in whiteBalance_values):
            whiteBalance_url = f"http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=setConfig&VideoInOptions[{ChannelNo}].WhiteBalance={whiteBalance}"
            whiteBalance_request = requests.get(whiteBalance_url, auth=HTTPDigestAuth(self.user, self.password))
            return Status(StatusCode.OK)
        else:
            return Status(StatusCode.INVALID_ARGUMENT, 
                    why = f"Unsupported saturation value! Received: {whiteBalance} | supported values: {whiteBalance_values}")

    def get_whiteBalance(self):
        whiteBalance_values = {"Disable", "Auto", "Custom", "Sunny", "Cloudy", "Home", "Office", "Night"}
        whiteBalance_url = f"http://{self.ip}:{self.http_port}/cgi-bin/configManager.cgi?action=getConfig&name=VideoInOptions"
        whiteBalance_request = requests.get(whiteBalance_url, auth=HTTPDigestAuth(self.user, self.password)).content.decode("utf-8")
        whiteBalance__index = 1
        notFund = -1
        for whiteBalance_value in whiteBalance_values:
            if (whiteBalance_request.split()[whiteBalance__index].find(whiteBalance_value) != notFund):
                return whiteBalance_value

    def set_resolution(self, resolution):
        resolution_values = {(1920.0,1080.0):0,(640.0,480.0):1}
        stream_id = resolution_values.get((resolution.width, resolution.height),None)
        status_code = Status(StatusCode.INVALID_ARGUMENT,
            why="Unsupported resolution value! Received: ({},{}) | Supported: {}".format(resolution.width,resolution.height,list(resolution_values.keys())))
        return (self.set_stream_channel_id(stream_id) if stream_id is not None else status_code)

    def get_resolution(self):
        resolution = Resolution()
        width, height = self.video_channel.cap.get(
            cv2.CAP_PROP_FRAME_WIDTH), self.video_channel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resolution.width, resolution.height = int(width), int(height)
        self.resolution = resolution
        return resolution

    def set_stream_channel_id(self, stream_id):
        if stream_id < 0 or stream_id > 1:
            return Status(StatusCode.INVALID_ARGUMENT,why="Invalid stream channel id")
        else:
            self.video_channel.release()
            self.stream_channel = stream_id
            return self.__configure_stream()

    def get_stream_channel_id(self):
        return self.stream_channel

if __name__ == '__main__':
    import json
    import time
    import time
    config_path = '../etc/conf/config.json'
    config = json.load(open(config_path, 'r'))
    camera_config = config['camera']

    print('---RUNNING EXAMPLE DEMO OF THE CAMERA DRIVER---')
    print(f'camera configurations on path: {config_path}')

    camera = IntelBDriver(camera_config)
    while True:
        img = camera.get_np_image()
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
