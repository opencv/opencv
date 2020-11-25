import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.*;
import org.opencv.3d.Cv3d;
import org.opencv.calib.Calib;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


class PerspectiveCorrectionRun {
    void perspectiveCorrection (String[] args) {
        String img1Path = args[0], img2Path = args[1];
        Mat img1 = Imgcodecs.imread(img1Path);
        Mat img2 = Imgcodecs.imread(img2Path);

        //! [find-corners]
        MatOfPoint2f corners1 = new MatOfPoint2f(), corners2 = new MatOfPoint2f();
        boolean found1 = Calib.findChessboardCorners(img1, new Size(9, 6), corners1 );
        boolean found2 = Calib.findChessboardCorners(img2, new Size(9, 6), corners2 );
        //! [find-corners]

        if (!found1 || !found2) {
            System.out.println("Error, cannot find the chessboard corners in both images.");
            System.exit(-1);
        }

        //! [estimate-homography]
        Mat H = new Mat();
        H = Cv3d.findHomography(corners1, corners2);
        System.out.println(H.dump());
        //! [estimate-homography]

        //! [warp-chessboard]
        Mat img1_warp = new Mat();
        Imgproc.warpPerspective(img1, img1_warp, H, img1.size());
        //! [warp-chessboard]

        Mat img_draw_warp = new Mat();
        List<Mat> list1 = new ArrayList<>(), list2 = new ArrayList<>() ;
        list1.add(img2);
        list1.add(img1_warp);
        Core.hconcat(list1, img_draw_warp);
        HighGui.imshow("Desired chessboard view / Warped source chessboard view", img_draw_warp);

        //! [compute-transformed-corners]
        Mat img_draw_matches = new Mat();
        list2.add(img1);
        list2.add(img2);
        Core.hconcat(list2, img_draw_matches);
        Point []corners1Arr = corners1.toArray();

        for (int i = 0 ; i < corners1Arr.length; i++) {
            Mat pt1 = new Mat(3, 1, CvType.CV_64FC1), pt2 = new Mat();
            pt1.put(0, 0, corners1Arr[i].x, corners1Arr[i].y, 1 );

            Core.gemm(H, pt1, 1, new Mat(), 0, pt2);
            double[] data = pt2.get(2, 0);
            Core.divide(pt2, new Scalar(data[0]), pt2);

            double[] data1 =pt2.get(0, 0);
            double[] data2 = pt2.get(1, 0);
            Point end = new Point((int)(img1.cols()+ data1[0]), (int)data2[0]);
            Imgproc.line(img_draw_matches, corners1Arr[i], end, RandomColor(), 2);
        }

        HighGui.imshow("Draw matches", img_draw_matches);
        HighGui.waitKey(0);
        //! [compute-transformed-corners]

        System.exit(0);
    }

    Scalar RandomColor () {
        Random rng = new Random();
        int r = rng.nextInt(256);
        int g = rng.nextInt(256);
        int b = rng.nextInt(256);
        return new Scalar(r, g, b);
    }
}

public class PerspectiveCorrection {
    public static void main (String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new PerspectiveCorrectionRun().perspectiveCorrection(args);
    }
}
