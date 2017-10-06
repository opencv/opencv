/**
 * @file CopyMakeBorder.java
 * @brief Sample code that shows the functionality of copyMakeBorder
 */

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.Random;

class CopyMakeBorderRun {

    public void run(String[] args) {

        //! [variables]
        // Declare the variables
        Mat src, dst = new Mat();
        int top, bottom, left, right;
        int borderType = Core.BORDER_CONSTANT;
        String window_name = "copyMakeBorder Demo";
        Random rng;
        //! [variables]

        //! [load]
        String imageName = ((args.length > 0) ? args[0] : "../data/lena.jpg");

        // Load an image
        src = Imgcodecs.imread(imageName, Imgcodecs.IMREAD_COLOR);

        // Check if image is loaded fine
        if( src.empty() ) {
            System.out.println("Error opening image!");
            System.out.println("Program Arguments: [image_name -- default ../data/lena.jpg] \n");
            System.exit(-1);
        }
        //! [load]

        // Brief how-to for this program
        System.out.println("\n" +
                "\t copyMakeBorder Demo: \n" +
                "\t -------------------- \n" +
                " ** Press 'c' to set the border to a random constant value \n" +
                " ** Press 'r' to set the border to be replicated \n" +
                " ** Press 'ESC' to exit the program \n");

        //![create_window]
        HighGui.namedWindow( window_name, HighGui.WINDOW_AUTOSIZE );
        //![create_window]

        //! [init_arguments]
        // Initialize arguments for the filter
        top = (int) (0.05*src.rows()); bottom = top;
        left = (int) (0.05*src.cols()); right = left;
        //! [init_arguments]

        while( true ) {
            //! [update_value]
            rng = new Random();
            Scalar value = new Scalar( rng.nextInt(256),
                    rng.nextInt(256), rng.nextInt(256) );
            //! [update_value]

            //! [copymakeborder]
            Core.copyMakeBorder( src, dst, top, bottom, left, right, borderType, value);
            //! [copymakeborder]
            //! [display]
            HighGui.imshow( window_name, dst );
            //! [display]

            //![check_keypress]
            char c = (char) HighGui.waitKey(500);
            c = Character.toLowerCase(c);

            if( c == 27 )
            { break; }
            else if( c == 'c' )
            { borderType = Core.BORDER_CONSTANT;}
            else if( c == 'r' )
            { borderType = Core.BORDER_REPLICATE;}
            //![check_keypress]
        }

        System.exit(0);
    }
}

public class CopyMakeBorder {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new CopyMakeBorderRun().run(args);
    }
}
