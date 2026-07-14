import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class PyramidsRun {

    String window_name = "Pyramids Demo";

    public void run(String[] args) {
        /// General instructions
        System.out.println("\n" +
                " Zoom In-Out demo    \n" +
                "------------------   \n" +
                " * [i] -> Zoom [i]n  \n" +
                " * [o] -> Zoom [o]ut \n" +
                " * [ESC] -> Close program \n");

        //! [load]
        String filename = ((args.length > 0) ? args[0] : "../data/chicky_512.png");

        // Load the image
        Mat src = Imgcodecs.imread(filename);

        // Check if image is loaded fine
        if( src.empty() ) {
            System.out.println("Error opening image!");
            System.out.println("Program Arguments: [image_name -- default ../data/chicky_512.png] \n");
            System.exit(-1);
        }
        //! [load]

        //! [loop]
        while (true){
            //! [show_image]
            HighGui.imshow( window_name, src );
            //! [show_image]
            char c = (char) HighGui.waitKey(0);
            c = Character.toLowerCase(c);

            if( c == 27 ){
                break;
                //![pyrup]
            }else if( c == 'i'){
                Imgproc.pyrUp( src, src, new Size( src.cols()*2, src.rows()*2 ) );
                System.out.println( "** Zoom In: Image x 2" );
                //![pyrup]
                //![pyrdown]
            }else if( c == 'o'){
                Imgproc.pyrDown( src, src, new Size( src.cols()/2, src.rows()/2 ) );
                System.out.println( "** Zoom Out: Image / 2" );
                //![pyrdown]
            }
        }
        //! [loop]

        System.exit(0);
    }
}

public class Pyramids {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new PyramidsRun().run(args);
    }
}
