import cv2
import os

input_path = "input/big_video.mp4"
output_folder = "temp_clips"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(input_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frames_per_clip = fps * 5   # 5 second clips

clip_count = 0
frame_count = 0
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frames_per_clip == 0:
        if out:
            out.release()

        clip_path = f"{output_folder}/clip_{clip_count}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(clip_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        clip_count += 1

    out.write(frame)
    frame_count += 1

if out:
    out.release()

cap.release()
print("Clips created successfully.")