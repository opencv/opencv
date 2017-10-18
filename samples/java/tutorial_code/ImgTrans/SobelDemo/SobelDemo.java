/**
 * @file SobelDemo.java
 * @brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
 */

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class SobelDemoRun {

    public void run(String[] args) {

        //! [declare_variables]
        // First we declare the variables we are going to use
        Mat src, src_gray = new Mat();
        Mat grad = new Mat();
        String window_name = "Sobel Demo - Simple Edge Detector";
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        //! [declare_variables]

        //! [load]
        // As usual we load our source image (src)
        // Check number of arguments
        if (args.length == 0){
            System.out.println("Not enough parameters!");
            System.out.println("Program Arguments: [image_path]");
            System.exit(-1);
        }

        // Load the image
        src = Imgcodecs.imread(args[0]);

        // Check if image is loaded fine
        if( src.empty() ) {
            System.out.println("Error opening image: " + args[0]);
            System.exit(-1);
        }
        //! [load]

        //! [reduce_noise]
        // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        Imgproc.GaussianBlur( src, src, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT );
        //! [reduce_noise]

        //! [convert_to_gray]
        // Convert the image to grayscale
        Imgproc.cvtColor( src, src_gray, Imgproc.COLOR_RGB2GRAY );
        //! [convert_to_gray]

        //! [sobel]
        /// Generate grad_x and grad_y
        Mat grad_x = new Mat(), grad_y = new Mat();
        Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

        /// Gradient X
        //Imgproc.Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, Core.BORDER_DEFAULT );
        Imgproc.Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, Core.BORDER_DEFAULT );

        /// Gradient Y
        //Imgproc.Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, Core.BORDER_DEFAULT );
        Imgproc.Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, Core.BORDER_DEFAULT );
        //! [sobel]

        //![convert]
        // converting back to CV_8U
        Core.convertScaleAbs( grad_x, abs_grad_x );
        Core.convertScaleAbs( grad_y, abs_grad_y );
        //![convert]

        //! [add_weighted]
        /// Total Gradient (approximate)
        Core.addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        //! [add_weighted]

        //! [display]
        HighGui.imshow( window_name, grad );
        HighGui.waitKey(0);
        //! [display]

        System.exit(0);
    }
}

public class SobelDemo {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new SobelDemoRun().run(args);
    }
}
