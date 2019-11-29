import cv2
import numpy as np


class SoccerBallTracker:

    def __init__(self):
        self.frameNumber = 0
        self.objectnessThreshold = 0.5  # Objectness threshold
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        self.inpWidth = 416  # Width of network's input image
        self.inpHeight = 416  # Height of network's input image
        self.object_left = 0
        self.object_top = 0
        self.object_width = 0
        self.object_height = 0

        classes_file = "./coco.names"
        self.classes = None
        with open(classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        model_configuration = "yolov3.cfg"
        model_weights = "yolov3.weights"

        self.net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)

    def render_with_tracking_info(self, img):
        return img

    def detect_yolo(self, img):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                if detection[4] > self.objectnessThreshold:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    if self.classes[classId] == "sports ball":
                        confidence = scores[classId]
                        if confidence > self.confThreshold:
                            center_x = int(detection[0] * frame_width)
                            center_y = int(detection[1] * frame_height)
                            width = int(detection[2] * frame_width)
                            height = int(detection[3] * frame_height)
                            left = int(center_x - width / 2)
                            top = int(center_y - height / 2)
                            classIds.append(classId)
                            confidences.append(float(confidence))
                            boxes.append([left, top, width, height])
        if len(boxes) > 0:
            return boxes[np.argmax(np.array(confidences))]
        return None

    def detect_and_track(self, img):
        box = self.detect_yolo(img)
        self.frameNumber += 1
        if box is not None:
            return box
        # TODO add tracking
        return [0, 0, 0, 0]

    def return_new_bounding_box(self, frame):
        self.detect_and_track(frame)
        right = 0
        bottom = 0
        return (self.object_left, self.object_top), (right, bottom)

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


if __name__ == "__main__":
    soccer_tracker = SoccerBallTracker()

    cap = cv2.VideoCapture('soccer-ball.mp4')
    while True:
        ret, frame = cap.read()
        if frame is not None:
            left, top, width, height = soccer_tracker.detect_and_track(frame)
            result = cv2.rectangle(frame, (left, top), (left+width, top+height), (255, 178, 50), 3)
            cv2.imshow('frame', result)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
