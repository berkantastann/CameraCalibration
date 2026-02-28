import cv2
import yaml
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calibration.yaml")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0=crop more, 1=keep more FOV (may show black borders)")
    args = ap.parse_args()

    with open(args.calib, "r") as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)

    cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # Use actual capture resolution (important!)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read frame.")
    h, w = frame.shape[:2]

    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

    print("ROI:", roi, " | Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Optional crop to ROI
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undist_crop = undist[y:y+rh, x:x+rw]
        else:
            undist_crop = undist

        cv2.imshow("Original", frame)
        cv2.imshow("Undistorted (crop ROI)", undist_crop)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()