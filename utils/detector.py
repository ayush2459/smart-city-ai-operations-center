from ultralytics import YOLO
import torch
import cv2

torch.set_num_threads(2)

class Detector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.to("cpu")

        # 🔥 Warmup model (prevents first-frame lag)
        dummy = torch.zeros((1, 3, 320, 320))
        self.model(dummy)

    def detect(self, frame):

        frame = cv2.resize(frame, (640, 384))

        with torch.no_grad():
            results = self.model(
                frame,
                imgsz=320,
                conf=0.25,
                classes=[0,2,3,5,7],
                verbose=False
            )[0]

        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

        return results, boxes