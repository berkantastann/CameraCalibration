import cv2
import glob
import os
import argparse
import numpy as np
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="calib_images/*.png")
    ap.add_argument("--pattern_rows", type=int, required=True,
                    help="number of inner corners per column (rows)")
    ap.add_argument("--pattern_cols", type=int, required=True,
                    help="number of inner corners per row (cols)")
    ap.add_argument("--square_size", type=float, required=True,
                    help="square size in your chosen unit (e.g., 0.024 for 24mm)")
    ap.add_argument("--out", default="calibration.yaml")
    args = ap.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)  # (cols, rows)

    images = sorted(glob.glob(args.images))
    if not images:
        raise RuntimeError(f"No images found: {args.images}")

    # 3D points for one view (0,0,0), (1,0,0), ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    good = 0
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        vis = img.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            good += 1
            cv2.drawChessboardCorners(vis, pattern_size, corners2, found)

        cv2.putText(vis, f"{os.path.basename(path)} | found={found}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if found else (0,0,255), 2)
        cv2.imshow("Corner detection", vis)
        cv2.waitKey(80)

    cv2.destroyAllWindows()

    if good < 10:
        raise RuntimeError(f"Too few valid images ({good}). Capture more / improve views.")

    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Compute mean reprojection error
    total_err = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        n = len(projected)
        total_err += err * err
        total_points += n

    mean_error = np.sqrt(total_err / total_points)

    print("\n=== Calibration Results ===")
    print("RMS (OpenCV ret):", ret)
    print("Mean reprojection error (px):", mean_error)
    print("Camera matrix K:\n", K)
    print("Dist coeffs:\n", dist.ravel())
    print("Image size:", img_size)
    print("Valid images used:", good, "/", len(images))

    # Save
    data = {
        "image_width": int(img_size[0]),
        "image_height": int(img_size[1]),
        "pattern_cols": int(args.pattern_cols),
        "pattern_rows": int(args.pattern_rows),
        "square_size": float(args.square_size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rms": float(ret),
        "mean_reprojection_error_px": float(mean_error),
    }

    with open(args.out, "w") as f:
        yaml.safe_dump(data, f)

    print("\nSaved:", args.out)

if __name__ == "__main__":
    main()