import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

//This program demonstrates how to use OpenCV PCA to extract the orientation of an object.
class IntroductionToPCA {
    private void drawAxis(Mat img, Point p_, Point q_, Scalar colour, float scale) {
        Point p = new Point(p_.x, p_.y);
        Point q = new Point(q_.x, q_.y);
        //! [visualization1]
        double angle = Math.atan2(p.y - q.y, p.x - q.x); // angle in radians
        double hypotenuse = Math.sqrt((p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));

        // Here we lengthen the arrow by a factor of scale
        q.x = (int) (p.x - scale * hypotenuse * Math.cos(angle));
        q.y = (int) (p.y - scale * hypotenuse * Math.sin(angle));
        Imgproc.line(img, p, q, colour, 1, Core.LINE_AA, 0);

        // create the arrow hooks
        p.x = (int) (q.x + 9 * Math.cos(angle + Math.PI / 4));
        p.y = (int) (q.y + 9 * Math.sin(angle + Math.PI / 4));
        Imgproc.line(img, p, q, colour, 1, Core.LINE_AA, 0);

        p.x = (int) (q.x + 9 * Math.cos(angle - Math.PI / 4));
        p.y = (int) (q.y + 9 * Math.sin(angle - Math.PI / 4));
        Imgproc.line(img, p, q, colour, 1, Core.LINE_AA, 0);
        //! [visualization1]
    }

    private double getOrientation(MatOfPoint ptsMat, Mat img) {
        List<Point> pts = ptsMat.toList();
        //! [pca]
        // Construct a buffer used by the pca analysis
        int sz = pts.size();
        Mat dataPts = new Mat(sz, 2, CvType.CV_64F);
        double[] dataPtsData = new double[(int) (dataPts.total() * dataPts.channels())];
        for (int i = 0; i < dataPts.rows(); i++) {
            dataPtsData[i * dataPts.cols()] = pts.get(i).x;
            dataPtsData[i * dataPts.cols() + 1] = pts.get(i).y;
        }
        dataPts.put(0, 0, dataPtsData);

        // Perform PCA analysis
        Mat mean = new Mat();
        Mat eigenvectors = new Mat();
        Mat eigenvalues = new Mat();
        Core.PCACompute2(dataPts, mean, eigenvectors, eigenvalues);
        double[] meanData = new double[(int) (mean.total() * mean.channels())];
        mean.get(0, 0, meanData);

        // Store the center of the object
        Point cntr = new Point(meanData[0], meanData[1]);

        // Store the eigenvalues and eigenvectors
        double[] eigenvectorsData = new double[(int) (eigenvectors.total() * eigenvectors.channels())];
        double[] eigenvaluesData = new double[(int) (eigenvalues.total() * eigenvalues.channels())];
        eigenvectors.get(0, 0, eigenvectorsData);
        eigenvalues.get(0, 0, eigenvaluesData);
        //! [pca]

        //! [visualization]
        // Draw the principal components
        Imgproc.circle(img, cntr, 3, new Scalar(255, 0, 255), 2);
        Point p1 = new Point(cntr.x + 0.02 * eigenvectorsData[0] * eigenvaluesData[0],
                cntr.y + 0.02 * eigenvectorsData[1] * eigenvaluesData[0]);
        Point p2 = new Point(cntr.x - 0.02 * eigenvectorsData[2] * eigenvaluesData[1],
                cntr.y - 0.02 * eigenvectorsData[3] * eigenvaluesData[1]);
        drawAxis(img, cntr, p1, new Scalar(0, 255, 0), 1);
        drawAxis(img, cntr, p2, new Scalar(255, 255, 0), 5);

        double angle = Math.atan2(eigenvectorsData[1], eigenvectorsData[0]); // orientation in radians
        //! [visualization]

        return angle;
    }

    public void run(String[] args) {
        //! [pre-process]
        // Load image
        String filename = args.length > 0 ? args[0] : "../data/pca_test1.jpg";
        Mat src = Imgcodecs.imread(filename);

        // Check if image is loaded successfully
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }

        Mat srcOriginal = src.clone();
        HighGui.imshow("src", srcOriginal);

        // Convert image to grayscale
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        // Convert image to binary
        Mat bw = new Mat();
        Imgproc.threshold(gray, bw, 50, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        //! [pre-process]

        //! [contours]
        // Find all the contours in the thresholded image
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bw, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        for (int i = 0; i < contours.size(); i++) {
            // Calculate the area of each contour
            double area = Imgproc.contourArea(contours.get(i));
            // Ignore contours that are too small or too large
            if (area < 1e2 || 1e5 < area)
                continue;

            // Draw each contour only for visualisation purposes
            Imgproc.drawContours(src, contours, i, new Scalar(0, 0, 255), 2);
            // Find the orientation of each shape
            getOrientation(contours.get(i), src);
        }
        //! [contours]

        HighGui.imshow("output", src);

        HighGui.waitKey();
        System.exit(0);
    }
}

public class IntroductionToPCADemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new IntroductionToPCA().run(args);
    }
}
