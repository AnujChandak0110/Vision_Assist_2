"""Integration test: camera + YOLO + Gemini API."""
import cv2
import sys
import time

print("=" * 50)
print("TEST 1: CAMERA")
print("=" * 50)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ok, frame = cap.read()
if ok:
    print(f"PASS  frame shape={frame.shape}, dtype={frame.dtype}")
else:
    print("FAIL  could not read frame from camera")
    sys.exit(1)
cap.release()

print()
print("=" * 50)
print("TEST 2: YOLO DETECTION ON LIVE FRAME")
print("=" * 50)
from ultralytics import YOLO
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Give camera a moment to warm up
for _ in range(5):
    cap.read()
ok, frame = cap.read()
cap.release()

model = YOLO("yolov8n.pt")
resized = cv2.resize(frame, (416, 416))
results = model.predict(source=resized, imgsz=416, conf=0.4, verbose=False, device="cpu")
boxes = results[0].boxes if results else []
print(f"PASS  {len(boxes)} detection(s) in current frame")
for box in list(boxes)[:5]:
    cls_id = int(box.cls.item())
    label = results[0].names.get(cls_id, str(cls_id))
    conf = float(box.conf.item())
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    print(f"       -> {label}  conf={conf:.2f}  bbox=({x1},{y1},{x2},{y2})")

print()
print("=" * 50)
print("TEST 3: GEMINI API (gemini-2.0-flash)")
print("=" * 50)
from llm.cloud_api import call_cloud_api, DEFAULT_GEMINI_MODEL
import os
print(f"Model in code : {DEFAULT_GEMINI_MODEL}")
print(f"Model in .env : {os.getenv('GEMINI_MODEL', 'not set')}")

t0 = time.time()
result = call_cloud_api("person at center, near; chair at left, medium", "scene_description_accuracy", "scene_description")
elapsed = time.time() - t0
if result and len(result) > 5:
    print(f"PASS  ({elapsed:.1f}s): \"{result}\"")
else:
    print(f"FAIL  got empty/short response in {elapsed:.1f}s: \"{result}\"")

print()
print("TEST 3b: GEMINI NAVIGATION MODE")
t0 = time.time()
result2 = call_cloud_api("car at center, near", "low_confidence", "navigation")
elapsed2 = time.time() - t0
if result2 and len(result2) > 5:
    print(f"PASS  ({elapsed2:.1f}s): \"{result2}\"")
else:
    print(f"FAIL  ({elapsed2:.1f}s): \"{result2}\"")

print()
print("=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
