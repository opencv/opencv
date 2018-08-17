import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.List;

class GeometricDrawingRun{

    private static final int W = 400;

    public void run(){
        //! [create_images]
        /// Windows names
        String atom_window = "Drawing 1: Atom";
        String rook_window = "Drawing 2: Rook";

        /// Create black empty images
        Mat atom_image = Mat.zeros( W, W, CvType.CV_8UC3 );
        Mat rook_image = Mat.zeros( W, W, CvType.CV_8UC3 );
        //! [create_images]

        //! [draw_atom]
        /// 1. Draw a simple atom:
        /// -----------------------
        MyEllipse( atom_image, 90.0 );
        MyEllipse( atom_image, 0.0 );
        MyEllipse( atom_image, 45.0 );
        MyEllipse( atom_image, -45.0 );

        /// 1.b. Creating circles
        MyFilledCircle( atom_image, new Point( W/2, W/2) );
        //! [draw_atom]

        //! [draw_rook]
        /// 2. Draw a rook
        /// ------------------
        /// 2.a. Create a convex polygon
        MyPolygon( rook_image );

        //! [rectangle]
        /// 2.b. Creating rectangles
        Imgproc.rectangle( rook_image,
                new Point( 0, 7*W/8 ),
                new Point( W, W),
                new Scalar( 0, 255, 255 ),
                -1,
                8,
                0 );
        //! [rectangle]

        /// 2.c. Create a few lines
        MyLine( rook_image, new Point( 0, 15*W/16 ), new Point( W, 15*W/16 ) );
        MyLine( rook_image, new Point( W/4, 7*W/8 ), new Point( W/4, W ) );
        MyLine( rook_image, new Point( W/2, 7*W/8 ), new Point( W/2, W ) );
        MyLine( rook_image, new Point( 3*W/4, 7*W/8 ), new Point( 3*W/4, W ) );
        //! [draw_rook]

        /// 3. Display your stuff!
        HighGui.imshow( atom_window, atom_image );
        HighGui.moveWindow( atom_window, 0, 200 );
        HighGui.imshow( rook_window, rook_image );
        HighGui.moveWindow( rook_window, W, 200 );

        HighGui.waitKey( 0 );
        System.exit(0);
    }

    /// Function Declaration

    /**
     * @function MyEllipse
     * @brief Draw a fixed-size ellipse with different angles
     */
    //! [my_ellipse]
    private void MyEllipse( Mat img, double angle ) {
        int thickness = 2;
        int lineType = 8;
        int shift = 0;

        Imgproc.ellipse( img,
                new Point( W/2, W/2 ),
                new Size( W/4, W/16 ),
                angle,
                0.0,
                360.0,
                new Scalar( 255, 0, 0 ),
                thickness,
                lineType,
                shift );
    }
    //! [my_ellipse]
    /**
     * @function MyFilledCircle
     * @brief Draw a fixed-size filled circle
     */
    //! [my_filled_circle]
    private void MyFilledCircle( Mat img, Point center ) {
        int thickness = -1;
        int lineType = 8;
        int shift = 0;

        Imgproc.circle( img,
                center,
                W/32,
                new Scalar( 0, 0, 255 ),
                thickness,
                lineType,
                shift );
    }
    //! [my_filled_circle]
    /**
     * @function MyPolygon
     * @function Draw a simple concave polygon (rook)
     */
    //! [my_polygon]
    private void MyPolygon( Mat img ) {
        int lineType = 8;
        int shift = 0;

        /** Create some points */
        Point[] rook_points = new Point[20];
        rook_points[0]  = new Point(     W/4, 7*W/8   );
        rook_points[1]  = new Point(   3*W/4, 7*W/8   );
        rook_points[2]  = new Point(   3*W/4, 13*W/16 );
        rook_points[3]  = new Point( 11*W/16, 13*W/16 );
        rook_points[4]  = new Point( 19*W/32, 3*W/8   );
        rook_points[5]  = new Point(   3*W/4, 3*W/8   );
        rook_points[6]  = new Point(   3*W/4, W/8     );
        rook_points[7]  = new Point( 26*W/40, W/8     );
        rook_points[8]  = new Point( 26*W/40, W/4     );
        rook_points[9]  = new Point( 22*W/40, W/4     );
        rook_points[10] = new Point( 22*W/40, W/8     );
        rook_points[11] = new Point( 18*W/40, W/8     );
        rook_points[12] = new Point( 18*W/40, W/4     );
        rook_points[13] = new Point( 14*W/40, W/4     );
        rook_points[14] = new Point( 14*W/40, W/8     );
        rook_points[15] = new Point(     W/4, W/8     );
        rook_points[16] = new Point(     W/4, 3*W/8   );
        rook_points[17] = new Point( 13*W/32, 3*W/8   );
        rook_points[18] = new Point(  5*W/16, 13*W/16 );
        rook_points[19] = new Point(     W/4, 13*W/16 );

        MatOfPoint matPt = new MatOfPoint();
        matPt.fromArray(rook_points);

        List<MatOfPoint> ppt = new ArrayList<MatOfPoint>();
        ppt.add(matPt);

        Imgproc.fillPoly(img,
                ppt,
                new Scalar( 255, 255, 255 ),
                lineType,
                shift,
                new Point(0,0) );
    }
    //! [my_polygon]
    /**
     * @function MyLine
     * @brief Draw a simple line
     */
    //! [my_line]
    private void MyLine( Mat img, Point start, Point end ) {
        int thickness = 2;
        int lineType = 8;
        int shift = 0;

        Imgproc.line( img,
                start,
                end,
                new Scalar( 0, 0, 0 ),
                thickness,
                lineType,
                shift );
    }
    //! [my_line]
}

public class BasicGeometricDrawing {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new GeometricDrawingRun().run();
    }
}
