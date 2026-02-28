import cv2
import os
import time
import argparse

def list_cameras(max_devices=10):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib_images", help="output folder")
    ap.add_argument("--device", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cams = list_cameras()
    print("Available camera indices:", cams)
    print("Using device:", args.device)

    cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try different --device index.")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            time.sleep(0.1)
            continue

        disp = frame.copy()
        cv2.putText(disp, f"SPACE=save | q=quit | saved={idx}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture calibration images", disp)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
        if k == ord(' '):
            path = os.path.join(args.out, f"img_{idx:03d}.png")
            cv2.imwrite(path, frame)
            print("Saved:", path)
            idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()