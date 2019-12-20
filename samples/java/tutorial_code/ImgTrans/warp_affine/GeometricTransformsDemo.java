import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class GeometricTransforms {
    public void run(String[] args) {
        //! [Load the image]
        String filename = args.length > 0 ? args[0] : "../data/lena.jpg";
        Mat src = Imgcodecs.imread(filename);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }
        //! [Load the image]

        //! [Set your 3 points to calculate the  Affine Transform]
        Point[] srcTri = new Point[3];
        srcTri[0] = new Point( 0, 0 );
        srcTri[1] = new Point( src.cols() - 1, 0 );
        srcTri[2] = new Point( 0, src.rows() - 1 );

        Point[] dstTri = new Point[3];
        dstTri[0] = new Point( 0, src.rows()*0.33 );
        dstTri[1] = new Point( src.cols()*0.85, src.rows()*0.25 );
        dstTri[2] = new Point( src.cols()*0.15, src.rows()*0.7 );
        //! [Set your 3 points to calculate the  Affine Transform]

        //! [Get the Affine Transform]
        Mat warpMat = Imgproc.getAffineTransform( new MatOfPoint2f(srcTri), new MatOfPoint2f(dstTri) );
        //! [Get the Affine Transform]

        //! [Apply the Affine Transform just found to the src image]
        Mat warpDst = Mat.zeros( src.rows(), src.cols(), src.type() );

        Imgproc.warpAffine( src, warpDst, warpMat, warpDst.size() );
        //! [Apply the Affine Transform just found to the src image]

        /** Rotating the image after Warp */

        //! [Compute a rotation matrix with respect to the center of the image]
        Point center = new Point(warpDst.cols() / 2, warpDst.rows() / 2);
        double angle = -50.0;
        double scale = 0.6;
        //! [Compute a rotation matrix with respect to the center of the image]

        //! [Get the rotation matrix with the specifications above]
        Mat rotMat = Imgproc.getRotationMatrix2D( center, angle, scale );
        //! [Get the rotation matrix with the specifications above]

        //! [Rotate the warped image]
        Mat warpRotateDst = new Mat();
        Imgproc.warpAffine( warpDst, warpRotateDst, rotMat, warpDst.size() );
        //! [Rotate the warped image]

        //! [Show what you got]
        HighGui.imshow( "Source image", src );
        HighGui.imshow( "Warp", warpDst );
        HighGui.imshow( "Warp + Rotate", warpRotateDst );
        //! [Show what you got]

        //! [Wait until user exits the program]
        HighGui.waitKey(0);
        //! [Wait until user exits the program]

        System.exit(0);
    }
}

public class GeometricTransformsDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new GeometricTransforms().run(args);
    }
}
