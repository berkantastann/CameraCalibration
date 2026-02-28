# calib_gui.py
import cv2
import numpy as np
import yaml
import argparse
import time
from pathlib import Path

def list_cameras(max_devices=10):
    cams = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cams.append(i)
        cap.release()
    return cams

def build_object_points(cols, rows, square_size):
    # (cols, rows) are INNER CORNERS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp

def draw_hud(img, lines, ok=True):
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
        y += 26
    return out

def mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_err2, total_points = 0.0, 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        n = len(proj)
        total_err2 += err * err
        total_points += n
    return float(np.sqrt(total_err2 / max(total_points, 1)))

def save_yaml(path, w, h, cols, rows, square_size, K, dist, rms, mre, device, fps, used):
    data = {
        "camera": {
            "device_index": int(device),
            "resolution": {"width": int(w), "height": int(h)},
            "fps_estimate": float(fps),
        },
        "pattern": {
            "type": "chessboard",
            "inner_corners": {"cols": int(cols), "rows": int(rows)},
            "square_size": float(square_size),
            "unit": "meters"  
        },
        "calibration": {
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.tolist(),
            "rms": float(rms),
            "mean_reprojection_error_px": float(mre),
            "frames_used": int(used),
        }
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cols", type=int, required=True, help="inner corners (cols)")
    ap.add_argument("--rows", type=int, required=True, help="inner corners (rows)")
    ap.add_argument("--square", type=float, required=True, help="square size (meters recommended)")
    ap.add_argument("--out", default="camera.yaml")
    ap.add_argument("--alpha", type=float, default=0.0, help="undistort crop/FOV balance 0..1")
    args = ap.parse_args()

    cams = list_cameras()
    print("Available camera indices:", cams)
    print("Using device:", args.device)

    cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try different --device.")

    pattern_size = (args.cols, args.rows)
    objp_single = build_object_points(args.cols, args.rows, args.square)

    objpoints = []
    imgpoints = []

    K = None
    dist = None
    rms = None
    mre = None
    map1 = map2 = None
    newK = None
    roi = None

    undistort_on = False

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # FPS estimate
    t0 = time.time()
    frames = 0
    fps_est = 0.0

    print("\nControls:")
    print("  SPACE : capture frame (only if corners found)")
    print("  c     : calibrate (needs >= 10 frames)")
    print("  u     : toggle undistort preview (after calibration)")
    print("  s     : save YAML (after calibration)")
    print("  r     : reset collected frames")
    print("  q     : quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frames += 1
        if frames % 30 == 0:
            dt = time.time() - t0
            if dt > 0:
                fps_est = frames / dt

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        ok = True
        show = frame.copy()

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(show, pattern_size, corners2, found)
        else:
            ok = False

        if undistort_on and K is not None and map1 is not None:
            show_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            if roi is not None:
                x, y, rw, rh = roi
                if rw > 0 and rh > 0:
                    show_u = show_u[y:y+rh, x:x+rw]
            cv2.imshow("Undistorted", show_u)

        hud = [
            f"device={args.device}  res={w}x{h}  fps~{fps_est:.1f}",
            f"pattern inner corners colsxrows = {args.cols}x{args.rows}  square={args.square}",
            f"corners_found={found}  captured={len(imgpoints)}",
            "SPACE=capture  c=calibrate  u=undistort  s=save  r=reset  q=quit"
        ]
        if mre is not None:
            hud.append(f"calibrated: MRE={mre:.3f}px  RMS={rms:.3f}")

        show2 = draw_hud(show, hud, ok=ok)
        cv2.imshow("Calibration", show2)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        if k == ord('r'):
            objpoints.clear()
            imgpoints.clear()
            K = dist = None
            rms = mre = None
            map1 = map2 = None
            newK = None
            roi = None
            undistort_on = False
            if cv2.getWindowProperty("Undistorted", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Undistorted")
            print("Reset: cleared captured frames and calibration.")
            continue

        if k == ord(' ') and found:
            imgpoints.append(corners2)
            objpoints.append(objp_single.copy())
            print(f"Captured frame: {len(imgpoints)}")
            continue

        if k == ord('c'):
            if len(imgpoints) < 10:
                print("Need at least 10 good frames to calibrate.")
                continue

            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, (w, h), None, None
            )
            mre = mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

            print("\n=== Calibrated ===")
            print("RMS:", float(rms))
            print("Mean reprojection error (px):", mre)
            print("K:\n", K)
            print("dist:", dist.ravel())
            print("ROI:", roi)
            print("Press 'u' to preview undistortion, 's' to save.\n")
            continue

        if k == ord('u'):
            if K is None or map1 is None:
                print("Calibrate first (press 'c').")
            else:
                undistort_on = not undistort_on
                if not undistort_on:
                    if cv2.getWindowProperty("Undistorted", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Undistorted")
                print("Undistort preview:", "ON" if undistort_on else "OFF")
            continue

        if k == ord('s'):
            if K is None:
                print("Calibrate first (press 'c').")
                continue
            out_path = Path(args.out).resolve()
            save_yaml(
                str(out_path), w, h, args.cols, args.rows, args.square,
                K, dist, rms, mre, args.device, fps_est, used=len(imgpoints)
            )
            print("Saved:", out_path)
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()