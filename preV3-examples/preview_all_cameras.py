import cv2
import depthai as dai

class Viewer(dai.Node):
    def __run__(self, 
                right: dai.message.ImgFrame,
                left: dai.message.ImgFrame,
                color: dai.message.ImgFrame):
        if cv2.waitKey(1) == ord('q'): raise KeyboardInterrupt()

        cv2.imshow("right", right.getCvFrame())
        cv2.imshow("left", right.getCvFrame())
        cv2.imshow("color", color.getCvFrame())

pipeline = dai.Pipeline()
with pipeline:
    Viewer(
        dai.node.MonoCamera(),
        dai.node.MonoCamera(),
        dai.node.ColorCamera().video)
dai.run(pipeline)
