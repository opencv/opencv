import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Range;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class CompareHist {
    public void run(String[] args) {
        //! [Load three images with different environment settings]
        if (args.length != 3) {
            System.err.println("You must supply 3 arguments that correspond to the paths to 3 images.");
            System.exit(0);
        }
        Mat srcBase = Imgcodecs.imread(args[0]);
        Mat srcTest1 = Imgcodecs.imread(args[1]);
        Mat srcTest2 = Imgcodecs.imread(args[2]);
        if (srcBase.empty() || srcTest1.empty() || srcTest2.empty()) {
            System.err.println("Cannot read the images");
            System.exit(0);
        }
        //! [Load three images with different environment settings]

        //! [Convert to HSV]
        Mat hsvBase = new Mat(), hsvTest1 = new Mat(), hsvTest2 = new Mat();
        Imgproc.cvtColor( srcBase, hsvBase, Imgproc.COLOR_BGR2HSV );
        Imgproc.cvtColor( srcTest1, hsvTest1, Imgproc.COLOR_BGR2HSV );
        Imgproc.cvtColor( srcTest2, hsvTest2, Imgproc.COLOR_BGR2HSV );
        //! [Convert to HSV]

        //! [Convert to HSV half]
        Mat hsvHalfDown = hsvBase.submat( new Range( hsvBase.rows()/2, hsvBase.rows() - 1 ), new Range( 0, hsvBase.cols() - 1 ) );
        //! [Convert to HSV half]

        //! [Using 50 bins for hue and 60 for saturation]
        int hBins = 50, sBins = 60;
        int[] histSize = { hBins, sBins };

        // hue varies from 0 to 179, saturation from 0 to 255
        float[] ranges = { 0, 180, 0, 256 };

        // Use the 0-th and 1-st channels
        int[] channels = { 0, 1 };
        //! [Using 50 bins for hue and 60 for saturation]

        //! [Calculate the histograms for the HSV images]
        Mat histBase = new Mat(), histHalfDown = new Mat(), histTest1 = new Mat(), histTest2 = new Mat();

        List<Mat> hsvBaseList = Arrays.asList(hsvBase);
        Imgproc.calcHist(hsvBaseList, new MatOfInt(channels), new Mat(), histBase, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histBase, histBase, 1, 0, Core.NORM_L1);

        List<Mat> hsvHalfDownList = Arrays.asList(hsvHalfDown);
        Imgproc.calcHist(hsvHalfDownList, new MatOfInt(channels), new Mat(), histHalfDown, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histHalfDown, histHalfDown, 1, 0, Core.NORM_L1);

        List<Mat> hsvTest1List = Arrays.asList(hsvTest1);
        Imgproc.calcHist(hsvTest1List, new MatOfInt(channels), new Mat(), histTest1, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest1, histTest1, 1, 0, Core.NORM_L1);

        List<Mat> hsvTest2List = Arrays.asList(hsvTest2);
        Imgproc.calcHist(hsvTest2List, new MatOfInt(channels), new Mat(), histTest2, new MatOfInt(histSize), new MatOfFloat(ranges), false);
        Core.normalize(histTest2, histTest2, 1, 0, Core.NORM_L1);
        //! [Calculate the histograms for the HSV images]

        //! [Apply the histogram comparison methods]
        for( int compareMethod = 0; compareMethod < 6; compareMethod++ ) {
            double baseBase = Imgproc.compareHist( histBase, histBase, compareMethod );
            double baseHalf = Imgproc.compareHist( histBase, histHalfDown, compareMethod );
            double baseTest1 = Imgproc.compareHist( histBase, histTest1, compareMethod );
            double baseTest2 = Imgproc.compareHist( histBase, histTest2, compareMethod );

            System.out.println("Method " + compareMethod + " Perfect, Base-Half, Base-Test(1), Base-Test(2) : " + baseBase + " / " + baseHalf
                    + " / " + baseTest1 + " / " + baseTest2);
        }
        //! [Apply the histogram comparison methods]
    }
}

public class CompareHistDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new CompareHist().run(args);
    }
}
