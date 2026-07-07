#!/usr/bin/env python3

"""Code for the Nested ArUco Markers tutorial.

Usage:
  python3 aruco_nested_detection.py --step create
  python3 aruco_nested_detection.py --step detect
  python3 aruco_nested_detection.py --step pose
  python3 aruco_nested_detection.py --step custom-generate
  python3 aruco_nested_detection.py --step custom-detect
"""

import argparse


def create_marker():
    ## [nested_marker_create_py]
    import cv2 as cv
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_NESTED_10)
    marker = cv.aruco.generateImageMarkerNested(dictionary, 0, 1200)
    cv.imwrite("pair_0.png", marker)
    ## [nested_marker_create_py]


def detect_markers():
    import cv2 as cv
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_NESTED_10)

    ## [nested_marker_detect_py]
    cap = cv.VideoCapture(0)
    params = cv.aruco.DetectorParameters()
    params.detectNestedMarkers = True
    detector = cv.aruco.ArucoDetector(dictionary, params)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, rejected = detector.detectMarkers(frame)
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.imshow("nested markers", frame)
        if cv.waitKey(1) == 27:
            break
    ## [nested_marker_detect_py]


def estimate_pose():
    import cv2 as cv
    import numpy as np

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_NESTED_10)
    cap = cv.VideoCapture(0)
    params = cv.aruco.DetectorParameters()
    params.detectNestedMarkers = True
    detector = cv.aruco.ArucoDetector(dictionary, params)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from camera 0")

    # Replace these with calibrated values for metric pose accuracy.
    height, width = frame.shape[:2]
    focal = float(max(width, height))
    camera_matrix = np.array([[focal, 0.0, width * 0.5],
                              [0.0, focal, height * 0.5],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    ## [nested_marker_pose_py]
    import numpy as np

    side_length = 0.20  # printed outer side in meters
    outer_pts, inner_pts = cv.aruco.getNestedMarkerObjectPoints(dictionary, 0, side_length)
    board = cv.aruco.Board([outer_pts, inner_pts], dictionary, np.array([0, 1]))

    while True:
        corners, ids, rejected = detector.detectMarkers(frame)
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is not None:
            obj_points, img_points = board.matchImagePoints(corners, ids)

            if len(obj_points) >= 4:
                ok, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

                if ok:
                    cv.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, side_length * 0.5)

        cv.imshow("nested marker pose", frame)
        if cv.waitKey(1) == 27:
            break

        ok, frame = cap.read()
        if not ok:
            break
    ## [nested_marker_pose_py]


def custom_generate():
    ## [nested_custom_rules_py]
    import cv2 as cv
    import numpy as np
    from pathlib import Path

    N = 4
    BORDER = 1
    PX = 160

    def bits_from(rows):
        return np.array([[1 if c == "1" else 0 for c in row] for row in rows], np.uint8)

    BILLBOARD_MARKERS = [
        dict(outer=["1101", "0010", "0000", "1001"], inner=["1110", "1100", "1100", "0000"],
             block=(0, 1), quiet_ring_is_white=False),
        dict(outer=["0111", "0011", "1010", "0100"], inner=["0001", "1000", "1111", "0010"],
             block=(2, 0), quiet_ring_is_white=True),
        dict(outer=["0001", "1110", "1110", "1111"], inner=["0010", "1010", "1001", "1101"],
             block=(0, 2), quiet_ring_is_white=True),
    ]

    CONSTELLATION_MARKERS = [
        dict(outer=["0000", "0011", "0110", "1111"],
             children=[(["0111", "1000", "0000", "1101"], (3, 2)),
                       (["0111", "0011", "1011", "0011"], (3, 0)),
                       (["1111", "0101", "0101", "1100"], (0, 3))]),
        dict(outer=["0000", "1011", "1010", "1110"],
             children=[(["0001", "1110", "1001", "0010"], (0, 1)),
                       (["0110", "1100", "0100", "0011"], (2, 1)),
                       (["0110", "1110", "1010", "1111"], (3, 3))]),
        dict(outer=["1110", "0110", "0001", "1011"],
             children=[(["1010", "0001", "0110", "0110"], (2, 1)),
                       (["0111", "0110", "1110", "1010"], (1, 3)),
                       (["0111", "0110", "0111", "1101"], (0, 0))]),
    ]
    ## [nested_custom_rules_py]

    ## [nested_custom_render_py]
    def render_binary(bits, px=PX):
        full = np.zeros((N + 2 * BORDER, N + 2 * BORDER), np.uint8)
        full[BORDER:BORDER + N, BORDER:BORDER + N] = bits
        return np.kron(full, np.full((px, px), 255, np.uint8))


    def render_billboard_marker(outer_bits, block, quiet_ring_is_white, inner_image, px=PX, gap=0.25):
        image = render_binary(outer_bits, px)
        bx, by = block
        x0 = (BORDER + bx) * px
        y0 = (BORDER + by) * px
        image[y0:y0 + 2 * px, x0:x0 + 2 * px] = 255 if quiet_ring_is_white else 0

        child = inner_image.copy()
        if not quiet_ring_is_white:
            child = 255 - child
        gap_px = int(round(gap * px))
        side = 2 * px - 2 * gap_px
        child = cv.resize(child, (side, side), interpolation=cv.INTER_NEAREST)
        image[y0 + gap_px:y0 + gap_px + side, x0 + gap_px:x0 + gap_px + side] = child
        return image


    def render_constellation_marker(outer_bits, children, px=PX, gap=0.2):
        image = render_binary(outer_bits, px)
        gap_px = int(round(gap * px))
        side = px - 2 * gap_px

        for child_image, (hx, hy) in children:
            child = child_image.copy()
            child = cv.resize(child, (side, side), interpolation=cv.INTER_NEAREST)
            patch = child if outer_bits[hy, hx] == 1 else 255 - child
            x0 = (BORDER + hx) * px + gap_px
            y0 = (BORDER + hy) * px + gap_px
            image[y0:y0 + side, x0:x0 + side] = patch
        return image


    def make_billboard_dictionary():
        dictionary_images, sheet_items = [], []
        for spec in BILLBOARD_MARKERS:
            inner = render_binary(bits_from(spec["inner"]))
            outer = render_billboard_marker(bits_from(spec["outer"]), spec["block"],
                                            spec["quiet_ring_is_white"], inner)
            out_id = len(dictionary_images)
            dictionary_images += [outer, inner]
            sheet_items.append((outer, f"out {out_id} / in {out_id + 1}"))
        return dictionary_images, sheet_items


    def make_constellation_dictionary():
        dictionary_images, sheet_items = [], []
        for spec in CONSTELLATION_MARKERS:
            outer_bits = bits_from(spec["outer"])
            children = [(render_binary(bits_from(rows)), host) for rows, host in spec["children"]]
            outer = render_constellation_marker(outer_bits, children)
            out_id = len(dictionary_images)
            dictionary_images.append(outer)
            dictionary_images.extend(child for child, host in children)
            child_ids = ",".join(str(i) for i in range(out_id + 1, out_id + 1 + len(children)))
            sheet_items.append((outer, f"out {out_id} / in {child_ids}"))
        return dictionary_images, sheet_items


    examples_by_name = {
        "billboard": make_billboard_dictionary(),
        "constellation": make_constellation_dictionary(),
    }
    ## [nested_custom_render_py]

    ## [nested_custom_build_dictionary_py]
    def build_dictionary(images):
        rows = []
        for image in images:
            ratios = cv.aruco.Dictionary.getCellRatiosFromImage(image, N, BORDER)
            rows.append(cv.aruco.Dictionary.getRatioListFromCellRatios(ratios))
        return cv.aruco.Dictionary(np.concatenate(rows, axis=0), N, 0,
                                   cv.aruco.DICT_ENCODING_CELL_RATIO)
    ## [nested_custom_build_dictionary_py]

    ## [nested_custom_write_dictionary_py]
    def write_dictionary(dictionary, path):
        storage = cv.FileStorage(str(path), cv.FILE_STORAGE_WRITE)
        dictionary.writeDictionary(storage)
        storage.release()


    def save_sheet(images, path):
        cols, rows = 3, 1
        tile, pad, header = 300, 14, 34
        sheet = np.full((rows * (tile + header + pad) + pad, cols * (tile + pad) + pad),
                        255, np.uint8)

        for idx, (image, label) in enumerate(images):
            row, col = 0, idx
            x0 = pad + col * (tile + pad)
            y0 = pad + row * (tile + header + pad)
            sheet[y0 + header:y0 + header + tile, x0:x0 + tile] = cv.resize(
                image, (tile, tile), interpolation=cv.INTER_AREA)
            cv.putText(sheet, label, (x0, y0 + header - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.50, 0, 1, cv.LINE_AA)
        cv.imwrite(str(path), sheet)

    out_dir = Path("custom_nested_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (dictionary_images, sheet_items) in examples_by_name.items():
        dictionary = build_dictionary(dictionary_images)
        write_dictionary(dictionary, out_dir / f"{name}.yml")
        save_sheet(sheet_items, out_dir / f"{name}_sheet.png")
        for idx, (image, label) in enumerate(sheet_items):
            cv.imwrite(str(out_dir / f"{name}_marker_{idx}.png"), image)
    ## [nested_custom_write_dictionary_py]


def custom_detect():
    ## [nested_custom_detect_py]
    import cv2 as cv

    name = "billboard"  # or "constellation"
    directory = "custom_nested_output"

    storage = cv.FileStorage(f"{directory}/{name}.yml", cv.FILE_STORAGE_READ)
    dictionary = cv.aruco.Dictionary()
    if not storage.isOpened() or not dictionary.readDictionary(storage.root()):
        raise RuntimeError("dictionary not found or invalid")
    storage.release()

    params = cv.aruco.DetectorParameters()
    params.detectNestedMarkers = True
    params.detectInvertedMarker = True
    params.errorCorrectionRate = 0.0
    params.perspectiveRemovePixelPerCell = 20

    detector = cv.aruco.ArucoDetector(dictionary, params)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open camera 0")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        cv.imshow("custom nested markers", frame)
        if cv.waitKey(1) == 27:
            break
    ## [nested_custom_detect_py]


def main():
    parser = argparse.ArgumentParser(description="Code for the Nested ArUco Markers tutorial.")
    parser.add_argument("--step", choices=("create", "detect", "pose", "custom-generate", "custom-detect"),
                        default="create")
    args = parser.parse_args()

    if args.step == "create":
        create_marker()
    elif args.step == "detect":
        detect_markers()
    elif args.step == "pose":
        estimate_pose()
    elif args.step == "custom-generate":
        custom_generate()
    elif args.step == "custom-detect":
        custom_detect()


if __name__ == "__main__":
    main()
