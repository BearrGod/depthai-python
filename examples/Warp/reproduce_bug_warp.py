#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(512, 512)
camRgb.setVideoSize(400, 400)
camRgb.setInterleaved(False)
camRgb.setFps(8)
maxFrameSize = camRgb.getPreviewWidth() * camRgb.getPreviewHeight() * 3

# Warp preview frame 1
warp1 = pipeline.create(dai.node.Warp)
# Create a custom warp mesh
tl = dai.Point2f(20, 20)
tr = dai.Point2f(460, 20)
ml = dai.Point2f(100, 250)
mr = dai.Point2f(400, 250)
bl = dai.Point2f(20, 460)
br = dai.Point2f(460, 460)
warp1.setWarpMesh([tl,tr,ml,mr,bl,br], 2, 3)
WARP1_OUTPUT_FRAME_SIZE = (512,512)
warp1.setOutputSize(WARP1_OUTPUT_FRAME_SIZE)
warp1.setMaxOutputFrameSize(WARP1_OUTPUT_FRAME_SIZE[0] * WARP1_OUTPUT_FRAME_SIZE[1] * 3)
warp1.setHwIds([1])
warp1.setInterpolation(dai.node.Warp.Properties.Interpolation.BYPASS)

camRgb.preview.link(warp1.inputImage)
xout1 = pipeline.create(dai.node.XLinkOut)
xout1.setStreamName('out1')

xout2 = pipeline.create(dai.node.XLinkOut)
xout2.setStreamName('out2')
warp1.out.link(xout1.input)
camRgb.video.link(xout2.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the rgb frames from the output defined above
    q1 = device.getOutputQueue(name="out1", maxSize=8, blocking=False)
    q2 = device.getOutputQueue(name="out2", maxSize=8, blocking=False)

    while True:
        in1 = q1.get()
        in2 = q2.get()
        if in1 is not None:
            cv2.imshow("Warped preview warp", in1.getCvFrame())
        if in2 is not None:
            cv2.imshow("Warped preview video", in2.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
