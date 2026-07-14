import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

class PointPolygonTest {
    public void run() {
        /// Create an image
        int r = 100;
        Mat src = Mat.zeros(new Size(4 * r, 4 * r), CvType.CV_8U);

        /// Create a sequence of points to make a contour
        List<Point> vert = new ArrayList<>(6);
        vert.add(new Point(3 * r / 2, 1.34 * r));
        vert.add(new Point(1 * r, 2 * r));
        vert.add(new Point(3 * r / 2, 2.866 * r));
        vert.add(new Point(5 * r / 2, 2.866 * r));
        vert.add(new Point(3 * r, 2 * r));
        vert.add(new Point(5 * r / 2, 1.34 * r));

        /// Draw it in src
        for (int i = 0; i < 6; i++) {
            Imgproc.line(src, vert.get(i), vert.get((i + 1) % 6), new Scalar(255), 3);
        }

        /// Get the contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(src, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        /// Calculate the distances to the contour
        Mat rawDist = new Mat(src.size(), CvType.CV_32F);
        float[] rawDistData = new float[(int) (rawDist.total() * rawDist.channels())];
        for (int i = 0; i < src.rows(); i++) {
            for (int j = 0; j < src.cols(); j++) {
                rawDistData[i * src.cols() + j] = (float) Imgproc
                        .pointPolygonTest(new MatOfPoint2f(contours.get(0).toArray()), new Point(j, i), true);
            }
        }
        rawDist.put(0, 0, rawDistData);

        MinMaxLocResult res = Core.minMaxLoc(rawDist);
        double minVal = Math.abs(res.minVal);
        double maxVal = Math.abs(res.maxVal);

        /// Depicting the distances graphically
        Mat drawing = Mat.zeros(src.size(), CvType.CV_8UC3);
        byte[] drawingData = new byte[(int) (drawing.total() * drawing.channels())];
        for (int i = 0; i < src.rows(); i++) {
            for (int j = 0; j < src.cols(); j++) {
                if (rawDistData[i * src.cols() + j] < 0) {
                    drawingData[(i * src.cols() + j) * 3] =
                            (byte) (255 - Math.abs(rawDistData[i * src.cols() + j]) * 255 / minVal);
                } else if (rawDistData[i * src.cols() + j] > 0) {
                    drawingData[(i * src.cols() + j) * 3 + 2] =
                            (byte) (255 - rawDistData[i * src.cols() + j] * 255 / maxVal);
                } else {
                    drawingData[(i * src.cols() + j) * 3] = (byte) 255;
                    drawingData[(i * src.cols() + j) * 3 + 1] = (byte) 255;
                    drawingData[(i * src.cols() + j) * 3 + 2] = (byte) 255;
                }
            }
        }
        drawing.put(0, 0, drawingData);
        Imgproc.circle(drawing, res.maxLoc, (int)res.maxVal, new Scalar(255, 255, 255), 2, 8, 0);

        /// Show your results
        HighGui.imshow("Source", src);
        HighGui.imshow("Distance and inscribed circle", drawing);

        HighGui.waitKey();
        System.exit(0);
    }
}

public class PointPolygonTestDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new PointPolygonTest().run();
    }

}
