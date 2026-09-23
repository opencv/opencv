#!/usr/bin/env python

"""create_board_charuco.py
Create and save a ChArUco board image (CHARUCO_1 or CHARUCO_2 layout).

Usage examples:
  CHARUCO_1 (classic, markers inside white squares):
    python create_board_charuco.py -w=5 -h=7 -sl=100 -ml=60 -d=0 -o=charuco1.png

  CHARUCO_2 (full-cell markers, markerLength == squareLength):
    python create_board_charuco.py -w=5 -h=7 -sl=100 -d=0 -bt=1 -o=charuco2.png
"""

import argparse
import sys
import cv2 as cv


DICT_HELP = (
    "dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, DICT_4X4_1000=3, "
    "DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
    "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, "
    "DICT_7X7_50=12, DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, "
    "DICT_ARUCO_ORIGINAL=16"
)


def main():
    parser = argparse.ArgumentParser(description="Create a ChArUco board image", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-o", help="Output image file (default: res.png)", default="res.png", dest="output")
    parser.add_argument("-w", help="Number of squares in X direction", default=5, type=int)
    parser.add_argument("-h", help="Number of squares in Y direction", default=7, type=int)
    parser.add_argument("-sl", help="Square side length in pixels", default=100, type=int, dest="sl")
    parser.add_argument("-ml", help="Marker side length in pixels. Ignored for CHARUCO_2 (markerLength == squareLength)",
                        default=60, type=int, dest="ml")
    parser.add_argument("-d", help=DICT_HELP, default=0, type=int, dest="d")
    parser.add_argument("-m", help="Margins in pixels. Default: squareLength-markerLength for CHARUCO_1, 0 for CHARUCO_2",
                        default=None, type=int, dest="m")
    parser.add_argument("-bb", help="Number of bits in marker borders", default=1, type=int, dest="bb")
    parser.add_argument("-si", help="Show generated image", action="store_true", dest="si")
    parser.add_argument("-bt", help="Board type: 0=CHARUCO_1 (classic), 1=CHARUCO_2 (full-cell markers)",
                        default=0, type=int, dest="bt")

    args = parser.parse_args()

    if args.show_help:
        parser.print_help()
        sys.exit()

    is_charuco2 = (args.bt == cv.aruco.CHARUCO_2)
    square_len = args.sl
    marker_len = square_len if is_charuco2 else args.ml

    if args.m is not None:
        margins = args.m
    else:
        margins = 0 if is_charuco2 else square_len - marker_len

    dictionary = cv.aruco.getPredefinedDictionary(args.d)
    board_type = cv.aruco.CHARUCO_2 if is_charuco2 else cv.aruco.CHARUCO_1
    board = cv.aruco.CharucoBoard((args.w, args.h), float(square_len), float(marker_len), dictionary,
                                  None, board_type)

    image_size = (args.w * square_len + 2 * margins, args.h * square_len + 2 * margins)
    board_image = board.generateImage(image_size, marginSize=margins, borderBits=args.bb)

    if args.si:
        cv.imshow("board", board_image)
        cv.waitKey(0)

    cv.imwrite(args.output, board_image)
    print(f"Board saved to {args.output}")


if __name__ == "__main__":
    main()
