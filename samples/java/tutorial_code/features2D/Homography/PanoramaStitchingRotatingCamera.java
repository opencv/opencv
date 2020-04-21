import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.core.Range;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


class PanoramaStitchingRotatingCameraRun {
    void basicPanoramaStitching (String[] args) {
        String img1path = args[0], img2path = args[1];
        Mat img1 = new Mat(), img2 = new Mat();
        img1 = Imgcodecs.imread(img1path);
        img2 = Imgcodecs.imread(img2path);

        //! [camera-pose-from-Blender-at-location-1]
        Mat c1Mo = new Mat( 4, 4, CvType.CV_64FC1 );
        c1Mo.put(0 ,0 ,0.9659258723258972, 0.2588190734386444, 0.0, 1.5529145002365112,
                 0.08852133899927139, -0.3303661346435547, -0.9396926164627075, -0.10281121730804443,
                 -0.24321036040782928, 0.9076734185218811, -0.342020183801651, 6.130080699920654,
                 0, 0, 0, 1 );
        //! [camera-pose-from-Blender-at-location-1]

        //! [camera-pose-from-Blender-at-location-2]
        Mat c2Mo = new Mat( 4, 4, CvType.CV_64FC1 );
        c2Mo.put(0, 0, 0.9659258723258972, -0.2588190734386444, 0.0, -1.5529145002365112,
                 -0.08852133899927139, -0.3303661346435547, -0.9396926164627075, -0.10281121730804443,
                 0.24321036040782928, 0.9076734185218811, -0.342020183801651, 6.130080699920654,
                 0, 0, 0, 1);
        //! [camera-pose-from-Blender-at-location-2]

        //! [camera-intrinsics-from-Blender]
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
        cameraMatrix.put(0, 0, 700.0, 0.0, 320.0, 0.0, 700.0, 240.0, 0, 0, 1 );
        //! [camera-intrinsics-from-Blender]

        //! [extract-rotation]
        Range rowRange = new Range(0,3);
        Range colRange = new Range(0,3);
        //! [extract-rotation]

        //! [compute-rotation-displacement]
        //c1Mo * oMc2
        Mat R1 = new  Mat(c1Mo, rowRange, colRange);
        Mat R2 = new Mat(c2Mo, rowRange, colRange);
        Mat R_2to1 = new Mat();
        Core.gemm(R1, R2.t(), 1, new Mat(), 0, R_2to1 );
        //! [compute-rotation-displacement]

        //! [compute-homography]
        Mat tmp = new Mat(), H = new Mat();
        Core.gemm(cameraMatrix, R_2to1, 1, new Mat(), 0, tmp);
        Core.gemm(tmp, cameraMatrix.inv(), 1, new Mat(), 0, H);
        Scalar s = new Scalar(H.get(2, 2)[0]);
        Core.divide(H, s, H);
        System.out.println(H.dump());
        //! [compute-homography]

        //! [stitch]
        Mat img_stitch = new Mat();
        Imgproc.warpPerspective(img2, img_stitch, H, new Size(img2.cols()*2, img2.rows()) );
        Mat half = new Mat();
        half =  new Mat(img_stitch, new Rect(0, 0, img1.cols(), img1.rows()));
        img1.copyTo(half);
        //! [stitch]

        Mat img_compare = new Mat();
        Mat img_space = Mat.zeros(new Size(50, img1.rows()), CvType.CV_8UC3);
        List<Mat>list = new ArrayList<>();
        list.add(img1);
        list.add(img_space);
        list.add(img2);
        Core.hconcat(list, img_compare);

        HighGui.imshow("Compare Images", img_compare);
        HighGui.imshow("Panorama Stitching", img_stitch);
        HighGui.waitKey(0);
        System.exit(0);
    }
}

public class PanoramaStitchingRotatingCamera {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new PanoramaStitchingRotatingCameraRun().basicPanoramaStitching(args);
    }
}
