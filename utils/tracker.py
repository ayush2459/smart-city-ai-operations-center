import math

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}          # id → centroid
        self.previous_positions = {}  # id → previous centroid

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, detections):
        updated_objects = {}

        for box in detections:
            centroid = self._centroid(box)

            matched_id = None

            for obj_id, prev_centroid in self.objects.items():
                distance = math.dist(prev_centroid, centroid)
                if distance < 50:
                    matched_id = obj_id
                    break

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            updated_objects[matched_id] = centroid

        self.previous_positions = self.objects
        self.objects = updated_objects

        return self.objects