from ultralytics import YOLO

class Detector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):

        results = self.model(
            frame,
            imgsz=320,
            conf=0.25,
            verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []

        return results, boxes