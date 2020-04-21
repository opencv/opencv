/**
 * @file Morphology_3.java
 * @brief Use morphology transformations for extracting horizontal and vertical lines sample code
 */

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Morphology_3Run {

    public void run(String[] args) {

        //! [load_image]
        // Check number of arguments
        if (args.length == 0){
            System.out.println("Not enough parameters!");
            System.out.println("Program Arguments: [image_path]");
            System.exit(-1);
        }

        // Load the image
        Mat src = Imgcodecs.imread(args[0]);

        // Check if image is loaded fine
        if( src.empty() ) {
            System.out.println("Error opening image: " + args[0]);
            System.exit(-1);
        }

        // Show source image
        HighGui.imshow("src", src);
        //! [load_image]

        //! [gray]
        // Transform source image to gray if it is not already
        Mat gray = new Mat();

        if (src.channels() == 3)
        {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        }
        else
        {
            gray = src;
        }

        // Show gray image
        showWaitDestroy("gray" , gray);
        //! [gray]

        //! [bin]
        // Apply adaptiveThreshold at the bitwise_not of gray
        Mat bw = new Mat();
        Core.bitwise_not(gray, gray);
        Imgproc.adaptiveThreshold(gray, bw, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, -2);

        // Show binary image
        showWaitDestroy("binary" , bw);
        //! [bin]

        //! [init]
        // Create the images that will use to extract the horizontal and vertical lines
        Mat horizontal = bw.clone();
        Mat vertical = bw.clone();
        //! [init]

        //! [horiz]
        // Specify size on horizontal axis
        int horizontal_size = horizontal.cols() / 30;

        // Create structure element for extracting horizontal lines through morphology operations
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontal_size,1));

        // Apply morphology operations
        Imgproc.erode(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure);

        // Show extracted horizontal lines
        showWaitDestroy("horizontal" , horizontal);
        //! [horiz]

        //! [vert]
        // Specify size on vertical axis
        int vertical_size = vertical.rows() / 30;

        // Create structure element for extracting vertical lines through morphology operations
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,vertical_size));

        // Apply morphology operations
        Imgproc.erode(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, verticalStructure);

        // Show extracted vertical lines
        showWaitDestroy("vertical", vertical);
        //! [vert]

        //! [smooth]
        // Inverse vertical image
        Core.bitwise_not(vertical, vertical);
        showWaitDestroy("vertical_bit" , vertical);

        // Extract edges and smooth image according to the logic
        // 1. extract edges
        // 2. dilate(edges)
        // 3. src.copyTo(smooth)
        // 4. blur smooth img
        // 5. smooth.copyTo(src, edges)

        // Step 1
        Mat edges = new Mat();
        Imgproc.adaptiveThreshold(vertical, edges, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 3, -2);
        showWaitDestroy("edges", edges);

        // Step 2
        Mat kernel = Mat.ones(2, 2, CvType.CV_8UC1);
        Imgproc.dilate(edges, edges, kernel);
        showWaitDestroy("dilate", edges);

        // Step 3
        Mat smooth = new Mat();
        vertical.copyTo(smooth);

        // Step 4
        Imgproc.blur(smooth, smooth, new Size(2, 2));

        // Step 5
        smooth.copyTo(vertical, edges);

        // Show final result
        showWaitDestroy("smooth - final", vertical);
        //! [smooth]

        System.exit(0);
    }

    private void showWaitDestroy(String winname, Mat img) {
        HighGui.imshow(winname, img);
        HighGui.moveWindow(winname, 500, 0);
        HighGui.waitKey(0);
        HighGui.destroyWindow(winname);
    }
}

public class Morphology_3 {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new Morphology_3Run().run(args);
    }
}
