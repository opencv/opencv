import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class EqualizeHist {
    public void run(String[] args) {
        //! [Load image]
        String filename = args.length > 0 ? args[0] : "../data/lena.jpg";
        Mat src = Imgcodecs.imread(filename);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }
        //! [Load image]

        //! [Convert to grayscale]
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        //! [Convert to grayscale]

        //! [Apply Histogram Equalization]
        Mat dst = new Mat();
        Imgproc.equalizeHist( src, dst );
        //! [Apply Histogram Equalization]

        //! [Display results]
        HighGui.imshow( "Source image", src );
        HighGui.imshow( "Equalized Image", dst );
        //! [Display results]

        //! [Wait until user exits the program]
        HighGui.waitKey(0);
        //! [Wait until user exits the program]

        System.exit(0);
    }
}

public class EqualizeHistDemo {

    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new EqualizeHist().run(args);
    }

}
