"""Detect all available camera devices and print their index and resolution."""
import cv2

print("Scanning for cameras (index 0-5)...\n")
found = []
for idx in range(6):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW = faster init on Windows
    if not cap.isOpened():
        print(f"  [{idx}] Not available")
        continue
    ret, frame = cap.read()
    if ret and frame is not None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        name = f"Camera index {idx}"
        print(f"  [{idx}] FOUND — resolution={w}x{h}  fps={fps:.0f}  *** USE THIS INDEX ***")
        found.append(idx)
    else:
        print(f"  [{idx}] Opens but cannot read frames (may be in use)")
    cap.release()

print()
if found:
    print(f"Available camera indexes: {found}")
    print(f"Built-in webcam is usually index 0.")
    print(f"External USB webcam is usually index 1 (or higher).")
    print(f"\nSet CAMERA_INDEX=<number> in your .env file to select.")
else:
    print("No cameras found. Check USB connection and drivers.")
