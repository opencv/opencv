from __future__ import print_function
import argparse
import sys
from glob import glob
import cv2 as cv
import numpy as np

PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range


# Input From the camera is to be implemented

class QrSample:
    def __init__(self, args):
        self.fname = ''
        self.fext = ''
        self.input = args.input
        self.detect = args.detect
        self.out = args.out
        self.multi = args.multi

    def getQRModeString(self):
        msg1 = "multi" if self.multi else ""
        msg2 = "detector" if self.detect else "decoder"
        msg = "QR {:s} {:s}".format(msg1, msg2)
        return msg

    def drawFPS(self, result, fps):
        message = '{:.2f} FPS({:s})'.format(fps, self.getQRModeString())
        cv.putText(result, message, (20, 20), 1,
                   cv.FONT_HERSHEY_DUPLEX, (0, 0, 255))

    def drawQRCodeContours(self, image, cnt):
        if cnt.size != 0:
            rows, cols, _ = image.shape
            show_radius = (2.813 * rows) / cols if (rows >
                                                    cols) else (2.813 * cols) / rows
            contour_radius = show_radius * 0.4
            cv.drawContours(
                image, [cnt], 0, (0, 255, 0), int(
                    round(contour_radius)))
            tpl = cnt.reshape((-1, 2))
            for x in tuple(tpl.tolist()):
                color = (255, 0, 0)
                cv.circle(
                    image, tuple(x), int(
                        round(contour_radius)), color, -1)

    def drawQRCodeResults(self, result, points, decode_info, fps):
        n = len(points)
        if isinstance(decode_info, str):
            decode_info = [decode_info]
        if n > 0:
            for i in range(n):
                cnt = np.array(points[i]).reshape((-1, 1, 2)).astype(np.int32)
                self.drawQRCodeContours(result, cnt)
                msg = 'QR[{:d}]@{} :'.format(i, *(cnt.reshape(1, -1).tolist()))
                print (msg, end="")
                if len(decode_info) > i:
                    if decode_info[i]:
                        print("'", decode_info[i], "'")
                    else:
                        print("Can't decode QR code")
                else:
                    print("Decode information is not available (disabled)")
        else:
            print ("QRCode not  detected!")
        self.drawFPS(result, fps)

    def runQR(self, qrCode, inputimg):
        if not self.multi:
            if not self.detect:
                decode_info, points, _ = qrCode.detectAndDecode(inputimg)
                dec_info = decode_info
            else:
                _, points = qrCode.detect(inputimg)
                dec_info = []
        else:
            if not self.detect:
                _, decode_info, points, _ = qrCode.detectAndDecodeMulti(
                    inputimg)
                dec_info = decode_info
            else:
                _, points = qrCode.detectMulti(inputimg)
                dec_info = []
        if points is None:
            points = []
        return points, dec_info

    def DetectQRFrmImage(self, inputfile):
        inputimg = cv.imread(inputfile, cv.IMREAD_COLOR)
        qrCode = cv.QRCodeDetector()
        count = 10
        timer = cv.TickMeter()
        for _ in range(count):
            timer.start()
            points, decode_info = self.runQR(qrCode, inputimg)
            timer.stop()
        fps = count / timer.getTimeSec()
        print("FPS: ", fps)
        result = inputimg
        self.drawQRCodeResults(result, points, decode_info, fps)
        cv.imshow("QR", result)
        cv.waitKey(1)
        if self.out != '':
            outfile = self.fname + self.fext
            print ("Saving Result: ", outfile)
            cv.imwrite(outfile, result)

        print("Press any key to exit ...")
        cv.waitKey(0)
        print("Exit")


def main():
    parser = argparse.ArgumentParser(
        description='This program detects the QR-codes input images using OpenCV Library.')
    parser.add_argument(
        '-i',
        '--input',
        help="input image path (default input file path is 'opencv_extra/testdata/cv/qrcode/multiple/*_qrcodes.png",
        default="../../../opencv_extra/testdata/cv/qrcode/multiple/*_qrcodes.png",
        metavar="")
    parser.add_argument(
        '-d',
        '--detect',
        help="detect QR code only (skip decoding) (default value is False)",
        action='store_true')
    parser.add_argument(
        '-m',
        '--multi',
        help="used to disable multiple qr-codes detection and enable single qr detection",
        action='store_false')
    parser.add_argument(
        '-o',
        '--out',
        help="path to result file (default output filename is qr_code.png)",
        default="qr_code.png",
        metavar="")
    args = parser.parse_args()
    qrinst = QrSample(args)
    if args.out != '':
        index = args.out.rfind('.')
        if index != -1:
            qrinst.fname = args.out[:index]
            qrinst.fext = args.out[index:]
        else:
            qrinst.fname = args.out
            qrinst.fext = ".png"
    for fn in glob(args.input):
        qrinst.DetectQRFrmImage(fn)


if __name__ == '__main__':
    main()
