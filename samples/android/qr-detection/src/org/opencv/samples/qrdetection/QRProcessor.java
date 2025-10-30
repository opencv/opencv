package org.opencv.samples.qrdetection;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.GraphicalCodeDetector;
import org.opencv.objdetect.QRCodeDetector;
import org.opencv.objdetect.QRCodeDetectorAruco;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

public class QRProcessor {
    private GraphicalCodeDetector detector;
    private static final String TAG = "QRProcessor";
    private Scalar LineColor = new Scalar(255, 0, 0);
    private Scalar FontColor = new Scalar(0, 0, 255);

    public QRProcessor(boolean useArucoDetector) {
        if (useArucoDetector)
            detector = new QRCodeDetectorAruco();
        else
            detector = new QRCodeDetector();
    }

    private boolean findQRs(Mat inputFrame, List<String> decodedInfo, MatOfPoint points,
                           boolean tryDecode, boolean multiDetect) {
        boolean result = false;
        if (multiDetect) {
            if (tryDecode)
                result = detector.detectAndDecodeMulti(inputFrame, decodedInfo, points);
            else
                result = detector.detectMulti(inputFrame, points);
        }
        else {
            if(tryDecode) {
                String s = detector.detectAndDecode(inputFrame, points);
                result = !points.empty();
                if (result)
                    decodedInfo.add(s);
            }
            else {
                result = detector.detect(inputFrame, points);
            }
        }
        return result;
    }

    private void renderQRs(Mat inputFrame, List<String> decodedInfo, MatOfPoint points) {
        for (int i = 0; i < points.rows(); i++) {
            for (int j = 0; j < points.cols(); j++) {
                Point pt1 = new Point(points.get(i, j));
                Point pt2 = new Point(points.get(i, (j + 1) % 4));
                Imgproc.line(inputFrame, pt1, pt2, LineColor, 3);
            }
            if (!decodedInfo.isEmpty()) {
                String decode = decodedInfo.get(i);
                if (decode.length() > 15) {
                    decode = decode.substring(0, 12) + "...";
                }
                int baseline[] = {0};
                Size textSize = Imgproc.getTextSize(decode, Imgproc.FONT_HERSHEY_COMPLEX, .95, 3, baseline);
                Scalar sum = Core.sumElems(points.row(i));
                Point start = new Point(sum.val[0] / 4. - textSize.width / 2., sum.val[1] / 4. - textSize.height / 2.);
                Imgproc.putText(inputFrame, decode, start, Imgproc.FONT_HERSHEY_COMPLEX, .95, FontColor, 3);
            }
        }
    }

    /* this method to be called from the outside. It processes the frame to find QR codes. */
    public synchronized Mat handleFrame(Mat inputFrame, boolean tryDecode, boolean multiDetect) {
        List<String> decodedInfo = new ArrayList<String>();
        MatOfPoint points = new MatOfPoint();
        boolean result = findQRs(inputFrame, decodedInfo, points, tryDecode, multiDetect);
        if (result) {
            renderQRs(inputFrame, decodedInfo, points);
        }
        points.release();
        return inputFrame;
    }
}
