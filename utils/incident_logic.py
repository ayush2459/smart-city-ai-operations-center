import numpy as np
import time


class IncidentDetector:

    def __init__(self):
        # Tracking memory
        self.previous_positions = {}
        self.previous_time = {}
        self.previous_speed = {}
        self.stopped_frames = {}

        self.frame_count = 0

        # Adjustable thresholds
        self.iou_threshold = 0.03
        self.stop_frame_threshold = 30
        self.aggressive_mode = False

        # Speed configuration
        self.fps = 25  # Set this to your real FPS
        self.speed_scale = 0.05  # Pixel-to-speed scaling (tune this)

    # ---------------------------------
    # IOU CALCULATION
    # ---------------------------------
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    # ---------------------------------
    # MAIN INCIDENT CHECK
    # ---------------------------------
    def check_incidents(self, positions, boxes):

        self.frame_count += 1
        incidents = {}

        position_keys = list(positions.keys())
        max_len = min(len(position_keys), len(boxes))

        if max_len == 0:
            return incidents

        # ===============================
        # COLLISION DETECTION
        # ===============================
        for i in range(max_len):
            for j in range(i + 1, max_len):

                iou = self.calculate_iou(boxes[i], boxes[j])
                threshold = self.iou_threshold

                if self.aggressive_mode:
                    threshold *= 0.7

                # Center distance
                cx1 = (boxes[i][0] + boxes[i][2]) / 2
                cy1 = (boxes[i][1] + boxes[i][3]) / 2
                cx2 = (boxes[j][0] + boxes[j][2]) / 2
                cy2 = (boxes[j][1] + boxes[j][3]) / 2

                center_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                if iou > threshold or center_dist < 20:
                    tracker_id = position_keys[i]

                    incidents[tracker_id] = {
                        "type": "collision",
                        "reason": f"Overlap {iou:.2f} | Distance {center_dist:.1f}",
                        "confidence": min(1.0, iou + 0.3),
                        "debug_iou": round(iou, 3)
                    }

        # ===============================
        # SPEED + STOP DETECTION
        # ===============================
        current_time = time.time()

        for i in range(max_len):

            tracker_id = position_keys[i]
            box = boxes[i]

            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            if tracker_id in self.previous_positions:

                px, py = self.previous_positions[tracker_id]

                # Pixel distance
                distance = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                # Time difference
                if tracker_id in self.previous_time:
                    time_diff = current_time - self.previous_time[tracker_id]
                else:
                    time_diff = 1 / self.fps

                if time_diff <= 0:
                    time_diff = 1 / self.fps

                # Raw pixel speed
                pixel_speed = distance / time_diff

                # Convert to scaled speed (approx km/h)
                speed = pixel_speed * self.speed_scale

                # Smooth speed
                prev_speed = self.previous_speed.get(tracker_id, speed)
                alpha = 0.6
                smoothed_speed = alpha * speed + (1 - alpha) * prev_speed

                self.previous_speed[tracker_id] = smoothed_speed

                # STOP detection
                if smoothed_speed < 3:
                    self.stopped_frames[tracker_id] = self.stopped_frames.get(tracker_id, 0) + 1
                else:
                    self.stopped_frames[tracker_id] = 0

                stop_threshold = self.stop_frame_threshold
                if self.aggressive_mode:
                    stop_threshold = int(stop_threshold * 0.7)

                if self.stopped_frames[tracker_id] > stop_threshold:
                    incidents[tracker_id] = {
                        "type": "stopped",
                        "reason": f"Vehicle stopped for {self.stopped_frames[tracker_id]} frames",
                        "confidence": 0.9,
                        "speed": round(smoothed_speed, 2)
                    }

            # Store state
            self.previous_positions[tracker_id] = (cx, cy)
            self.previous_time[tracker_id] = current_time

        return incidents