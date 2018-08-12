import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class CalcHist {
    public void run(String[] args) {
        //! [Load image]
        String filename = args.length > 0 ? args[0] : "../data/lena.jpg";
        Mat src = Imgcodecs.imread(filename);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }
        //! [Load image]

        //! [Separate the image in 3 places ( B, G and R )]
        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(src, bgrPlanes);
        //! [Separate the image in 3 places ( B, G and R )]

        //! [Establish the number of bins]
        int histSize = 256;
        //! [Establish the number of bins]

        //! [Set the ranges ( for B,G,R) )]
        float[] range = {0, 256}; //the upper boundary is exclusive
        MatOfFloat histRange = new MatOfFloat(range);
        //! [Set the ranges ( for B,G,R) )]

        //! [Set histogram param]
        boolean accumulate = false;
        //! [Set histogram param]

        //! [Compute the histograms]
        Mat bHist = new Mat(), gHist = new Mat(), rHist = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), new Mat(), bHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(1), new Mat(), gHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(2), new Mat(), rHist, new MatOfInt(histSize), histRange, accumulate);
        //! [Compute the histograms]

        //! [Draw the histograms for B, G and R]
        int histW = 512, histH = 400;
        int binW = (int) Math.round((double) histW / histSize);

        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );
        //! [Draw the histograms for B, G and R]

        //! [Normalize the result to ( 0, histImage.rows )]
        Core.normalize(bHist, bHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(gHist, gHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(rHist, rHist, 0, histImage.rows(), Core.NORM_MINMAX);
        //! [Normalize the result to ( 0, histImage.rows )]

        //! [Draw for each channel]
        float[] bHistData = new float[(int) (bHist.total() * bHist.channels())];
        bHist.get(0, 0, bHistData);
        float[] gHistData = new float[(int) (gHist.total() * gHist.channels())];
        gHist.get(0, 0, gHistData);
        float[] rHistData = new float[(int) (rHist.total() * rHist.channels())];
        rHist.get(0, 0, rHistData);

        for( int i = 1; i < histSize; i++ ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
        }
        //! [Draw for each channel]

        //! [Display]
        HighGui.imshow( "Source image", src );
        HighGui.imshow( "calcHist Demo", histImage );
        HighGui.waitKey(0);
        //! [Display]

        System.exit(0);
    }
}

public class CalcHistDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new CalcHist().run(args);
    }
}
