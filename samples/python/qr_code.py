# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# QR Code detection and decoding sample.
# Detects and decodes one or more QR codes from a webcam or image/video file.
#
# Usage:
#   qr_code.py [--input <path>]
#
# Examples:
#   python qr_code.py                        # webcam
#   python qr_code.py --input image.png      # image file
#   python qr_code.py --input video.mp4      # video file

import cv2
import argparse
import numpy as np

def draw_qr_results(frame, bboxes, data_list):
    for data, bbox in zip(data_list, bboxes):
        if bbox is not None and data:
            pts = bbox.astype(int).reshape(-1, 2)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            x, y = pts[0]
            cv2.putText(frame, data, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def process_frame(frame, detector):
    retval, decoded_list, bboxes, _ = detector.detectAndDecodeMulti(frame)
    if retval and bboxes is not None:
        draw_qr_results(frame, bboxes, decoded_list)
        for data in decoded_list:
            if data:
                print(f'Decoded: {data}')
    return frame 


def main():
    parser = argparse.ArgumentParser(description='QR Code Scanner - OpenCV Sample')
    parser.add_argument('--input', help='Path to image or video file. Uses webcam if not specified.')
    args = parser.parse_args()

    detector = cv2.QRCodeDetector()

    if args.input:
        # try as image first
        img = cv2.imread(args.input)
        if img is not None:
            result = process_frame(img, detector)
            cv2.imshow('QR Scanner', result)
            cv2.waitKey(0)
        else:
            # try as video
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                print(f'Error: Cannot open {args.input}')
                return
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = process_frame(frame, detector)
                cv2.imshow('QR Scanner', result)
                if cv2.waitKey(30) == 27:
                    break
            cap.release()
    else:
        print('Starting webcam... Press ESC to quit.')
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Error: Cannot open webcam.')
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = process_frame(frame, detector)
            cv2.putText(result, 'Press ESC to quit', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('QR Scanner', result)
            if cv2.waitKey(1) == 27:
                break
        cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
