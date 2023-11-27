package org.opencv.samples.qrdetection;

import org.opencv.core.Core;
import org.opencv.core.Mat;
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

public class QRdetectionProcessor {
    private GraphicalCodeDetector detector;
    private static final String TAG = "QRdetectionProcessor";
    private Scalar LineColor = new Scalar(255, 0, 0);
    private Scalar FontColor = new Scalar(0, 0, 255);
    public QRdetectionProcessor(boolean useArucoDetector) {
        if (useArucoDetector)
            detector = new QRCodeDetectorAruco();
        else
            detector = new QRCodeDetector();
    }

    public boolean findQRs(Mat inputFrame, List<String> decodedInfo, Mat points,
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

    /* this method to be called from the outside. It processes the frame to find QR codes. */
    public synchronized Mat qrFrame(Mat inputFrame, boolean tryDecode, boolean multiDetect) {
        List<String> decodedInfo = new ArrayList<String>();
        Mat points = new Mat();
        boolean result = findQRs(inputFrame, decodedInfo, points, tryDecode, multiDetect);
        if (result) {
            for (int i = 0; i < points.rows(); i++) {
                for (int j = 0; j < points.cols(); j++) {
                    Point pt1 = new Point(points.get(i, j));
                    Point pt2 = new Point(points.get(i, (j + 1) % 4));
                    Imgproc.line(inputFrame, pt1, pt2, LineColor, 3);
                }
                if (decodedInfo.size() == 0)
                    continue;
                String decode = decodedInfo.get(i);
                if (decode.length() > 15) {
                    decode = decode.substring(0, 12) + "...";
                }
                int baseline[]={0};
                Size textSize = Imgproc.getTextSize(decode, Imgproc.FONT_HERSHEY_COMPLEX, .95,3, baseline);
                Scalar sum = Core.sumElems(points.row(i));
                Point start = new Point(sum.val[0] / 4. - textSize.width / 2., sum.val[1] / 4. - textSize.height / 2.);
                Imgproc.putText(inputFrame, decode, start, Imgproc.FONT_HERSHEY_COMPLEX, .95, FontColor, 3);
            }
        }
        points.release();
        return inputFrame;
    }
}
