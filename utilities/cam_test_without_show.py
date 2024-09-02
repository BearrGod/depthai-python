#!/usr/bin/env python3

"""
Use 'T' to trigger autofocus, 'IOKL,.'
for manual exposure/focus:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O      1..33000 [us]
  sensitivity iso:   K   L    100..1600
  focus:             ,   .      0..255 [far..near]
To go back to auto controls:
  'E' - autoexposure
  'F' - autofocus (continuous)

Other controls:
'1' - AWB lock (true / false)
'2' - AE lock (true / false)
'3' - Select control: AWB mode
'4' - Select control: AE compensation
'5' - Select control: anti-banding/flicker mode
'6' - Select control: effect mode
'7' - Select control: brightness
'8' - Select control: contrast
'9' - Select control: saturation
'0' - Select control: sharpness
'[' - Select control: luma denoise
']' - Select control: chroma denoise
'\' - Select control: scene mode
';' - Select control: control mode
''' - Select control: capture intent
'a' 'd' - Increase/decrease dot projector intensity
'w' 's' - Increase/decrease flood LED intensity

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase

'/' to toggle printing camera settings: exposure, ISO, lens position, color temperature
"""

import depthai as dai
import os
from pathlib import Path
# os.environ["DEPTHAI_LEVEL"] = "debug"

import cv2
import numpy as np
import argparse
import collections
import time
from itertools import cycle
from pathlib import Path
from threading import Thread
import sys
import signal
from stress_test import stress_test, YOLO_LABELS, create_yolo

TEST_DURATION = 20 # the test will last 1 min

ALL_SOCKETS = ['rgb', 'left', 'right', 'cama', 'camb', 'camc', 'camd', 'came']

def socket_type_pair(arg):
    socket, type = arg.split(',')
    if not (socket in ALL_SOCKETS):
        raise ValueError("")
    if not (type in ['m', 'mono', 'c', 'color', 't', 'tof', 'th', 'thermal']):
        raise ValueError("")
    is_color = True if type in ['c', 'color'] else False
    is_tof = True if type in ['t', 'tof'] else False
    is_thermal = True if type in ['th', 'thermal'] else False
    return [socket, is_color, is_tof, is_thermal]


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-cams', '--cameras', type=socket_type_pair, nargs='+',
                    default=[],
                    help="Which camera sockets to enable, and type: c[olor] / m[ono] / t[of] / th[ermal]. "
                    "E.g: -cams rgb,m right,c . If not specified, all connected cameras will be used.")
parser.add_argument('-rp','--report-path',type=str,default="/tmp/som_report/",help="The directory in which to save the report of our diagnostic")
parser.add_argument('-mres', '--mono-resolution', type=int, default=800, choices={480, 400, 720, 800},
                    help="Select mono camera resolution (height). Default: %(default)s")
parser.add_argument('-cres', '--color-resolution', default='1080', choices={'720', '800', '1080', '1012', '1200', '1520', '4k', '5mp', '12mp', '13mp', '48mp'},
                    help="Select color camera resolution / height. Default: %(default)s")
parser.add_argument('-rot', '--rotate', const='all', choices={'all', 'rgb', 'mono'}, nargs="?",
                    help="Which cameras to rotate 180 degrees. All if not filtered")
parser.add_argument('-fps', '--fps', type=float, default=30,
                    help="FPS to set for all cameras")
parser.add_argument('-isp3afps', '--isp3afps', type=int, default=0,
                    help="3A FPS to set for all cameras")
parser.add_argument('-ds', '--isp-downscale', default=1, type=int,
                    help="Downscale the ISP output by this factor")
parser.add_argument('-rs', '--resizable-windows', action='store_true',
                    help="Make OpenCV windows resizable. Note: may introduce some artifacts")
parser.add_argument('-tun', '--camera-tuning', type=Path,
                    help="Path to custom camera tuning database")
parser.add_argument('-raw', '--enable-raw', default=False, action="store_true",
                    help='Enable the RAW camera streams')
parser.add_argument('-tofraw', '--tof-raw', action='store_true',
                    help="Show just ToF raw output instead of post-processed depth")
parser.add_argument('-tofint', '--tof-intensity', action='store_true',
                    help="Show also ToF intensity output alongside depth")
parser.add_argument('-tofamp', '--tof-amplitude', action='store_true',
                    help="Show also ToF amplitude output alongside depth")
parser.add_argument('-tofcm', '--tof-cm', action='store_true',
                    help="Show ToF depth output in centimeters, capped to 255")
parser.add_argument('-tofmedian', '--tof-median', choices=[0,3,5,7], default=5, type=int,
                    help="ToF median filter kernel size")
parser.add_argument('-rgbprev', '--rgb-preview', action='store_true',
                    help="Show RGB `preview` stream instead of full size `isp`")
parser.add_argument('-show', '--show-meta', action='store_true',
                    help="List frame metadata (seqno, timestamp, exp, iso etc). Can also toggle with `\`")

parser.add_argument('-d', '--device', default="", type=str,
                    help="Optional MX ID of the device to connect to.")

parser.add_argument('-ctimeout', '--connection-timeout', default=30000,
                    help="Connection timeout in ms. Default: %(default)s (sets DEPTHAI_CONNECTION_TIMEOUT environment variable)")

parser.add_argument('-btimeout', '--boot-timeout', default=30000,
                    help="Boot timeout in ms. Default: %(default)s (sets DEPTHAI_BOOT_TIMEOUT environment variable)")

parser.add_argument('-stress', action='store_true',
                    help="Run stress test. This will override all other options (except -d/--device) and will run a heavy pipeline until the user stops it.")

parser.add_argument("-stereo", action="store_true", default=False,
                    help="Create a stereo depth node if the device has a stereo pair.")

parser.add_argument("-yolo", type=str, default="",
                    help=f"Create a yolo detection network on the specified camera. E.g: -yolo cama. Available cameras: {ALL_SOCKETS}")

parser.add_argument("-gui", action="store_true",
                    help="Use GUI instead of CLI")
parser.add_argument("-h", "--help", action="store_true", default=False,
                    help="Show this help message and exit") # So you can forward --help to stress test, without it being consumed by cam_test.py

args = parser.parse_args()

# Set timeouts before importing depthai
os.environ["DEPTHAI_CONNECTION_TIMEOUT"] = str(args.connection_timeout)
os.environ["DEPTHAI_BOOT_TIMEOUT"] = str(args.boot_timeout)


if args.stress:
    stress_test(args.device)
    exit(0)

if args.help:
    parser.print_help()
    exit(0)

fps_registered : bool = False 
Path(args.report_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.report_path,"images/")).mkdir(parents=True, exist_ok=True)
report_file_handle = open(os.path.join(args.report_path,"som_report.log"),'w+')

if args.gui:
    import cam_test_gui
    cam_test_gui.main()

print("DepthAI version:", dai.__version__)
print("DepthAI path:", dai.__file__)
report_file_handle.write("DepthAI version: " +  dai.__version__ + "\n")
report_file_handle.write("DepthAI path: " +  dai.__file__ + "\n")

cam_socket_opts = {
    'rgb': dai.CameraBoardSocket.CAM_A,
    'left': dai.CameraBoardSocket.CAM_B,
    'right': dai.CameraBoardSocket.CAM_C,
    'cama': dai.CameraBoardSocket.CAM_A,
    'camb': dai.CameraBoardSocket.CAM_B,
    'camc': dai.CameraBoardSocket.CAM_C,
    'camd': dai.CameraBoardSocket.CAM_D,
    'came': dai.CameraBoardSocket.CAM_E,
}

rotate = {
    'rgb': args.rotate in ['all', 'rgb'],
    'left': args.rotate in ['all', 'mono'],
    'right': args.rotate in ['all', 'mono'],
    'cama': args.rotate in ['all', 'rgb'],
    'camb': args.rotate in ['all', 'mono'],
    'camc': args.rotate in ['all', 'mono'],
    'camd': args.rotate in ['all', 'rgb'],
    'came': args.rotate in ['all', 'mono'],
}

mono_res_opts = {
    400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
    720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
    800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
    1200: dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    '720':  dai.ColorCameraProperties.SensorResolution.THE_720_P,
    '800':  dai.ColorCameraProperties.SensorResolution.THE_800_P,
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '1012': dai.ColorCameraProperties.SensorResolution.THE_1352X1012,
    '1200': dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    '1520': dai.ColorCameraProperties.SensorResolution.THE_2024X1520,
    '4k':   dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '5mp': dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    '13mp': dai.ColorCameraProperties.SensorResolution.THE_13_MP,
    '48mp': dai.ColorCameraProperties.SensorResolution.THE_48_MP,
}



def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Calculates FPS over a moving window, configurable


class FPS:
    def __init__(self, window_size=60):
        self.dq = collections.deque(maxlen=window_size)
        self.fps = 0

    def update(self, timestamp=None):
        if timestamp == None:
            timestamp = time.monotonic()
        count = len(self.dq)
        if count > 0:
            self.fps = count / (timestamp - self.dq[0])
        self.dq.append(timestamp)

    def get(self):
        return self.fps

class Cycle:
    def __init__(self, enum_type, start_item=None):
        self.items = [item for name, item in vars(enum_type).items() if name.isupper()]
        # If start_item is provided, set the index to its position. Otherwise, default to 0
        self.index = self.items.index(start_item) if start_item else 0

    def step(self, n):
        self.index = (self.index + n) % len(self.items)
        return self.items[self.index]

    def next(self):
        return self.step(1)

    def prev(self):
        return self.step(-1)

def exit_cleanly(signum, frame):
    print("Exiting cleanly")
    cv2.destroyAllWindows()
    sys.exit(0)


def socket_to_socket_opt(socket: dai.CameraBoardSocket) -> str:
    return str(socket).split('.')[-1].replace("_", "").lower()


signal.signal(signal.SIGINT, exit_cleanly)
# Connect to device, so that we can get connected cameras in case of no args
success, device = dai.Device.getDeviceByMxId(args.device)
dai_device_args = []
if success:
    dai_device_args.append(device)
with dai.Device(*dai_device_args) as device:
    cam_list = []
    cam_type_color = {}
    cam_type_tof = {}
    cam_type_thermal = {}

    if not args.cameras:
        connected_cameras = device.getConnectedCameraFeatures()
        connected_imu = device.getConnectedIMU()
        device_speed  = device.getUsbSpeed()
        print("Connected Imu : ", connected_imu)
        print("Device Speed : " ,device_speed.name)
        report_file_handle.write("Device Connection Speed: " + device_speed.name + "\n")
        report_file_handle.write("Connected Imu: " + ("None" if (connected_imu =="") else connected_imu ) + "\n")
        args.cameras = [(socket_to_socket_opt(cam.socket), cam.supportedTypes[0] ==
                         dai.CameraSensorType.COLOR, cam.supportedTypes[0] == dai.CameraSensorType.TOF, cam.supportedTypes[0] == dai.CameraSensorType.THERMAL) for cam in connected_cameras]
        if not args.cameras:
            print("No cameras found!")
            report_file_handle.write("No cameras found! ")
            exit(1)

    print("Enabled cameras:")
    report_file_handle.write("Enabled cameras: \n") 
    enabed_cam_str = ""
    for socket, is_color, is_tof, is_thermal in args.cameras:
        cam_list.append(socket)
        cam_type_color[socket] = is_color
        cam_type_tof[socket] = is_tof
        cam_type_thermal[socket] = is_thermal
        print(socket.rjust(7), ':', 'tof' if is_tof else 'color' if is_color else 'thermal' if is_thermal else 'mono')
        report_file_handle.write(socket.rjust(7) +  ':' + (' tof' if is_tof else ' color' if is_color else ' thermal' if is_thermal else ' mono') + '\n')
    

    # Start defining a pipeline
    pipeline = dai.Pipeline()
    # Uncomment to get better throughput
    # pipeline.setXLinkChunkSize(0)

    control = pipeline.createXLinkIn()
    control.setStreamName('control')

    xinTofConfig = pipeline.createXLinkIn()
    xinTofConfig.setStreamName('tofConfig')

    cam = {}
    tof = {}
    xout = {}
    xout_raw = {}
    xout_tof_amp = {}
    xout_tof_int = {}
    streams = []
    yolo_passthrough_q_name = None
    for c in cam_list:
        print("CAM: ", c)
        tofEnableRaw = False
        xout[c] = pipeline.createXLinkOut()
        xout[c].setStreamName(c)
        streams.append(c)
        if cam_type_tof[c]:
            cam[c] = pipeline.create(dai.node.ColorCamera)  # .Camera
            if args.tof_raw:
                tofEnableRaw = True
            else:
                tof[c] = pipeline.create(dai.node.ToF)
                cam[c].raw.link(tof[c].input)
                tof[c].depth.link(xout[c].input)
                xinTofConfig.out.link(tof[c].inputConfig)
                if args.tof_median == 0:
                    tof[c].initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
                elif args.tof_median == 3:
                    tof[c].initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
                elif args.tof_median == 5:
                    tof[c].initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
                elif args.tof_median == 7:
                    tof[c].initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
                tofConfig = tof[c].initialConfig.get()  # TODO multiple instances
                if args.tof_amplitude:
                    amp_name = 'tof_amplitude_' + c
                    xout_tof_amp[c] = pipeline.create(dai.node.XLinkOut)
                    xout_tof_amp[c].setStreamName(amp_name)
                    streams.append(amp_name)
                    tof[c].amplitude.link(xout_tof_amp[c].input)
                if args.tof_intensity:
                    int_name = 'tof_intensity_' + c
                    xout_tof_int[c] = pipeline.create(dai.node.XLinkOut)
                    xout_tof_int[c].setStreamName(int_name)
                    streams.append(int_name)
                    tof[c].intensity.link(xout_tof_int[c].input)
        elif cam_type_thermal[c]:
            cam[c] = pipeline.create(dai.node.Camera)
            cam[c].setBoardSocket(cam_socket_opts[c])
            cam[c].setPreviewSize(256, 192)
            cam[c].raw.link(xout[c].input)
            xout_preview = pipeline.create(dai.node.XLinkOut)
            xout_preview.setStreamName('preview_' + c)
            cam[c].preview.link(xout_preview.input)
            streams.append('preview_' + c)
        elif cam_type_color[c]:
            cam[c] = pipeline.createColorCamera()
            cam[c].setResolution(color_res_opts[args.color_resolution])
            cam[c].setIspScale(1, args.isp_downscale)
            # cam[c].initialControl.setManualFocus(85) # TODO
            if args.rgb_preview:
                cam[c].preview.link(xout[c].input)
            else:
                cam[c].isp.link(xout[c].input)
            if args.yolo == c:
                yolo_passthrough_q_name, yolo_q_name = create_yolo(pipeline, cam[c])
                streams.append(yolo_q_name)
        else:
            cam[c] = pipeline.createMonoCamera()
            cam[c].setResolution(mono_res_opts[args.mono_resolution])
            cam[c].out.link(xout[c].input)
        cam[c].setBoardSocket(cam_socket_opts[c])
        # Num frames to capture on trigger, with first to be discarded (due to degraded quality)
        # cam[c].initialControl.setExternalTrigger(2, 1)
        # cam[c].initialControl.setStrobeExternal(48, 1)
        # cam[c].initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)

        # cam[c].initialControl.setManualExposure(15000, 400) # exposure [us], iso
        # When set, takes effect after the first 2 frames
        # cam[c].initialControl.setManualWhiteBalance(4000)  # light temperature in K, 1000..12000
        control.out.link(cam[c].inputControl)
        if rotate[c]:
            cam[c].setImageOrientation(
                dai.CameraImageOrientation.ROTATE_180_DEG)
        cam[c].setFps(args.fps)
        if args.isp3afps:
            cam[c].setIsp3aFps(args.isp3afps)

        if args.enable_raw or tofEnableRaw:
            raw_name = 'raw_' + c
            xout_raw[c] = pipeline.create(dai.node.XLinkOut)
            xout_raw[c].setStreamName(raw_name)
            streams.append(raw_name)
            cam[c].raw.link(xout_raw[c].input)
            cam[c].setRawOutputPacked(False)

    if args.camera_tuning:
        pipeline.setCameraTuningBlobPath(str(args.camera_tuning))

    stereo = None

    if args.stereo:
        try:
            try:
                calib = device.readCalibration2()
            except:
                raise Exception("Device is not calibrated.")
            eeprom = calib.getEepromData()
            left, right = eeprom.stereoRectificationData.leftCameraSocket, eeprom.stereoRectificationData.rightCameraSocket
            # Get the actual camera nodes
            # The cameras may have been specified with -cams rgb,c left,m right,m kind of names, so we need to handle these edge cases
            left_sock_opt = socket_to_socket_opt(left)
            right_sock_opt = socket_to_socket_opt(right)
            left_cam = cam.get(left_sock_opt, None)
            right_cam = cam.get(right_sock_opt, None)
            if not left_cam:
                if left == dai.CameraBoardSocket.CAM_A:
                    left_sock_opt = "rgb"
                elif left == dai.CameraBoardSocket.CAM_B:
                    left_sock_opt = "left"
                elif left == dai.CameraBoardSocket.CAM_C:
                    left_sock_opt = "right"
                left_cam = cam.get(left_sock_opt, None)
            if not right_cam:
                if right == dai.CameraBoardSocket.CAM_A:
                    right_sock_opt = "rgb"
                elif right == dai.CameraBoardSocket.CAM_B:
                    right_sock_opt = "left"
                elif right == dai.CameraBoardSocket.CAM_C:
                    right_sock_opt = "right"
                right_cam = cam.get(right_sock_opt, None)

            if left_cam and right_cam:
                cam_features = device.getConnectedCameraFeatures()
                left_cam_features = next(
                    filter(lambda c: c.socket == left, cam_features))
                right_cam_features = next(
                    filter(lambda c: c.socket == right, cam_features))
                if left_cam_features.width > 1280:
                    if args.isp_downscale == 1:
                        raise Exception(
                            "Can't create stereo depth with left cam width > 1280. Use --isp-downscale to downscale the image.")
                if right_cam_features.width > 1280:
                    if args.isp_downscale == 1:
                        raise Exception(
                            "Can't create stereo depth with right cam width > 1280. Use --isp-downscale to downscale the image.")
                left_out = "out"
                right_out = "out"
                if cam_type_color[left_sock_opt]:
                    left_out = "video"
                if cam_type_color[right_sock_opt]:
                    right_out = "video"

                print(
                    "Device is calibrated and has a stereo pair, creating StereoDepth node.")
                stereo = pipeline.createStereoDepth()
                stereo.setDefaultProfilePreset(
                    dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
                stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
                stereo.setLeftRightCheck(True)
                stereo.setSubpixel(True)
                stereo.setLeftRightCheck(True)
                getattr(left_cam, left_out).link(stereo.left)
                getattr(right_cam, right_out).link(stereo.right)
                xout_stereo = pipeline.createXLinkOut()
                depth_stream = "stereo_depth"
                xout_stereo.setStreamName(depth_stream)
                stereo.disparity.link(xout_stereo.input)
                streams.append(depth_stream)
            else:
                print("Couldn't create stereo depth node. Device has invalid calibration.")
        except Exception as e:
            print("Couldn't create depth:", e)

    # Pipeline is defined, now we can start it
    device.startPipeline(pipeline)

    print('Connected cameras:')
    report_file_handle.write("Connected cameras:\n")
    cam_name = {}
    for p in device.getConnectedCameraFeatures():
        print(
            f' -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:', end='')
        report_file_handle.write(f' -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:')
        print('auto ' if p.hasAutofocus else 'fixed', '- ', end='')
        report_file_handle.write('auto ' if p.hasAutofocus else 'fixed' + '- ')
        print(*[type.name for type in p.supportedTypes])
        for type in p.supportedTypes : 
            report_file_handle.write(type.name + ' ')
        report_file_handle.write('\n')
        cam_name[p.socket.name] = p.sensorName
        if args.enable_raw:
            cam_name['raw_'+p.socket.name] = p.sensorName
        if args.tof_amplitude:
            cam_name['tof_amplitude_'+p.socket.name] = p.sensorName
        if args.tof_intensity:
            cam_name['tof_intensity_'+p.socket.name] = p.sensorName

    print('USB speed:', device.getUsbSpeed().name)

    print('IR drivers:', device.getIrDrivers())

    q = {}
    fps_host = {}  # FPS computed based on the time we receive frames in app
    fps_capt = {}  # FPS computed based on capture timestamps from device
    for c in streams:
        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
        # The OpenCV window resize may produce some artifacts
        if args.resizable_windows:
            cv2.namedWindow(c, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(c, (640, 480))
        fps_host[c] = FPS()
        fps_capt[c] = FPS()

    controlQueue = device.getInputQueue('control')
    tofCfgQueue = device.getInputQueue('tofConfig')

    # Manual exposure/focus set step
    EXP_STEP = 500  # us
    ISO_STEP = 50
    LENS_STEP = 1 / 1024
    DOT_STEP = 0.05
    FLOOD_STEP = 0.05
    DOT_MAX = 1
    FLOOD_MAX = 1

    # Defaults and limits for manual focus/exposure controls
    lensPos = 0.59
    lensMin = 0.0
    lensMax = 1.0

    expTime = 20000
    expMin = 1
    expMax = 33000

    sensIso = 800
    sensMin = 100
    sensMax = 1600

    dotIntensity = 0
    floodIntensity = 0

    awb_mode = Cycle(dai.CameraControl.AutoWhiteBalanceMode)
    anti_banding_mode = Cycle(dai.CameraControl.AntiBandingMode)
    effect_mode = Cycle(dai.CameraControl.EffectMode)
    scene_mode = Cycle(dai.CameraControl.SceneMode)
    control_mode = Cycle(dai.CameraControl.ControlMode)
    capture_intent = Cycle(dai.CameraControl.CaptureIntent)

    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    control = 'none'
    show = args.show_meta

    jet_custom = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    jet_custom[0] = [0, 0, 0]

    print("Cam:", *['     ' + c.ljust(8)
                    for c in cam_list], "[host | capture timestamp]")
    cams_enabled_str = "Cam:" 
    for c in cam_list :
        cams_enabled_str = cams_enabled_str + '     ' + c.ljust(8)
    cams_enabled_str  = cams_enabled_str + "[host | capture timestamp] \n"
    report_file_handle.write(cams_enabled_str) 

    capture_list = streams.copy()
    capture_time = time.strftime('%Y%m%d_%H%M%S')
    times_since_last_saves = {}
    for c in streams : 
        times_since_last_saves[c] = 0
    yolo_passthrough_q = None
    if yolo_passthrough_q_name is not None:
        yolo_passthrough_q = device.getOutputQueue(yolo_passthrough_q_name, maxSize=1, blocking=False)

    begin_test = time.time()
    while time.time() - begin_test <= TEST_DURATION : #while True
        for c in streams:
            try:
                pkt = q[c].tryGet()
                capture_time = time.strftime('%Y%m%d_%H%M%S')
            except Exception as e:
                print(e)
                exit_cleanly(0, 0)
            if pkt is not None:
                fps_host[c].update()
                fps_capt[c].update(pkt.getTimestamp().total_seconds())
                if args.yolo and isinstance(pkt, dai.ImgDetections):
                    if yolo_passthrough_q is None:
                        continue
                    frame_pkt = yolo_passthrough_q.get()
                    frame = frame_pkt.getCvFrame()
                    if frame is None:
                        continue # No frame to draw on
                    for detection in pkt.detections:
                        bbox = np.array([detection.xmin * frame.shape[1], detection.ymin * frame.shape[0], detection.xmax * frame.shape[1], detection.ymax * frame.shape[0]], dtype=np.int32)
                        cv2.putText(frame, YOLO_LABELS[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.putText(frame, f"{int(detection.confidence)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    # cv2.imshow(c, frame)
                    continue
                width, height = pkt.getWidth(), pkt.getHeight()
                frame = pkt.getCvFrame()
                cam_skt = c.split('_')[-1]

                if c == "stereo_depth" and stereo is not None:
                    maxDisp = stereo.initialConfig.getMaxDisparity()
                    disp = (pkt.getCvFrame() * (255.0 / maxDisp)).astype(np.uint8)
                    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
                    # cv2.imshow(c, disp)
                    continue
                
                
                if cam_type_tof.get(cam_skt, None) and not (c.startswith('raw_') or c.startswith('tof_amplitude_') or c.startswith('tof_intensity_')):
                    if args.tof_cm:
                        # pixels represent `cm`, capped to 255. Value can be checked hovering the mouse
                        frame = (frame // 10).clip(0, 255).astype(np.uint8)
                    else:
                        max_depth = (tofConfig.phaseUnwrappingLevel + 1) * 1874 # 80MHz modulation freq. TODO slider
                        frame = np.interp(frame, (0, max_depth), (0, 255)).astype(np.uint8)
                        frame = cv2.applyColorMap(frame, jet_custom)
                elif cam_type_thermal[cam_skt] and c.startswith('cam'):
                    frame = frame.astype(np.float32)
                    frame = cv2.normalize(frame, frame, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)
                if show:
                    txt = f"[{c:5}, {pkt.getSequenceNum():4}, {pkt.getTimestamp().total_seconds():.6f}] "
                    txt += f"Exp: {pkt.getExposureTime().total_seconds()*1000:6.3f} ms, "
                    txt += f"ISO: {pkt.getSensitivity():4}, "
                    txt += f"Lens pos: {pkt.getLensPosition():3}, "
                    txt += f"Color temp: {pkt.getColorTemperature()} K"
                    if needs_newline:
                        print()
                        needs_newline = False
                    print(txt)
                capture = c in capture_list
                if capture:
                    capture_file_info = ('capture_' + c + '_' + cam_name[cam_socket_opts[cam_skt].name]
                         + '_' + str(width) + 'x' + str(height)
                         + '_' + capture_time
                         + '_exp_' + str(int(pkt.getExposureTime().total_seconds()*1e6))
                         + '_iso_' + str(pkt.getSensitivity())
                         + '_lens_' + str(pkt.getLensPosition())
                         + '_' + str(pkt.getColorTemperature()) + 'K'
                         + '_' + str(pkt.getSequenceNum())
                        )
                    # capture_list.remove(c)
                    print()
                if c.startswith('raw_') or c.startswith('tof_amplitude_') or c.startswith('tof_intensity_'):
                    if capture:
                        filename = capture_file_info + '_10bit.bw'
                        print('Saving:', filename)
                        frame.tofile(filename)
                    # Full range for display, use bits [15:6] of the 16-bit pixels
                    type = pkt.getType()
                    multiplier = 1
                    if type == dai.ImgFrame.Type.RAW10:
                        multiplier = (1 << (16-10))
                    if type == dai.ImgFrame.Type.RAW12:
                        multiplier = (1 << (16-4))
                    frame = frame * multiplier
                    # Debayer as color for preview/png
                    if cam_type_color[cam_skt]:
                        # See this for the ordering, at the end of page:
                        # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
                        # TODO add bayer order to ImgFrame getType()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BayerGB2BGR)
                else:
                    # Save YUV too, but only when RAW is also enabled (for tuning purposes)
                    if capture and args.enable_raw:
                        payload = pkt.getData()
                        filename = capture_file_info + '_P420.yuv'
                        print('Saving:', filename)
                        payload.tofile(filename)
                if capture and (time.time()- times_since_last_saves[c]) > 1  : # Save an image every second
                    filename = str(os.path.join(args.report_path,"images/")) + capture_file_info + '.png'
                    print('Saving:', filename)
                    cv2.imwrite(filename, frame)
                    times_since_last_saves[c] = time.time()
                #cv2.imshow(c, frame)
        print("\rFPS:",
              *["{:6.2f}|{:6.2f}".format(fps_host[c].get(),
                                         fps_capt[c].get()) for c in cam_list],
              end=' ', flush=True)
        needs_newline = True
        time.sleep(0.001) # sleep for 1ms since we don't want to idle the cpu
    
    report_file_handle.write("FPS:")
    for c in streams:
        report_file_handle.write("{:6.2f}|{:6.2f}".format(fps_host[c].get(),
                                         fps_capt[c].get()))
    report_file_handle.write("\n")
    report_file_handle.flush()
    report_file_handle.close()
    print()
