#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
stepSize = 0.05

newConfig = False

# Create pipeline
pipeline = dai.Pipeline()

tof = pipeline.create(dai.node.ToF)
camTof = pipeline.create(dai.node.Camera)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# ToF settings
camTof.setFps(30)
camTof.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
camTof.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
camTof.raw.link(tof.input)
tof.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)


spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (255, 255, 255)

    print("Use WASD keys to move ROI!")

    while True:
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        assert isinstance(inDepth, dai.ImgFrame)
        depthFrame = inDepth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialDataMsg = spatialCalcQueue.get()
        assert isinstance(spatialDataMsg, dai.SpatialLocationCalculatorData)
        spatialData = spatialDataMsg.getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
        # Show the frame
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        elif key == ord('1'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
            print('Switching calculation algorithm to MEAN!')
            newConfig = True
        elif key == ord('2'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
            print('Switching calculation algorithm to MIN!')
            newConfig = True
        elif key == ord('3'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MAX
            print('Switching calculation algorithm to MAX!')
            newConfig = True
        elif key == ord('4'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
            print('Switching calculation algorithm to MODE!')
            newConfig = True
        elif key == ord('5'):
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
            print('Switching calculation algorithm to MEDIAN!')
            newConfig = True

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            config.calculationAlgorithm = calculationAlgorithm
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
            newConfig = False