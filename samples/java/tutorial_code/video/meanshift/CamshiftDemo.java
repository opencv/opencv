import java.util.Arrays;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;


class Camshift {
    public void run(String[] args) {
        String filename = args[0];
        VideoCapture capture = new VideoCapture(filename);
        if (!capture.isOpened()) {
            System.out.println("Unable to open file!");
            System.exit(-1);
        }

        Mat frame = new Mat(), hsv_roi = new Mat(), mask = new Mat(), roi;

        // take the first frame of the video
        capture.read(frame);

        //setup initial location of window
        Rect track_window = new Rect(300, 200, 100, 50);

        // set up the ROI for tracking
        roi = new Mat(frame, track_window);
        Imgproc.cvtColor(roi, hsv_roi, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv_roi, new Scalar(0, 60, 32), new Scalar(180, 255, 255), mask);

        MatOfFloat range = new MatOfFloat(0, 256);
        Mat roi_hist = new Mat();
        MatOfInt histSize = new MatOfInt(180);
        MatOfInt channels = new MatOfInt(0);
        Imgproc.calcHist(Arrays.asList(hsv_roi), channels, mask, roi_hist, histSize, range);
        Core.normalize(roi_hist, roi_hist, 0, 255, Core.NORM_MINMAX);

        // Setup the termination criteria, either 10 iteration or move by at least 1 pt
        TermCriteria term_crit = new TermCriteria(TermCriteria.EPS | TermCriteria.COUNT, 10, 1);

        while (true) {
            Mat hsv = new Mat() , dst = new Mat();
            capture.read(frame);
            if (frame.empty()) {
                break;
            }
            Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
            Imgproc.calcBackProject(Arrays.asList(hsv), channels, roi_hist, dst, range, 1);

            // apply camshift to get the new location
            RotatedRect rot_rect = Video.CamShift(dst, track_window, term_crit);

            // Draw it on image
            Point[] points = new Point[4];
            rot_rect.points(points);
            for (int i = 0; i < 4 ;i++) {
                Imgproc.line(frame, points[i], points[(i+1)%4], new Scalar(255, 0, 0),2);
            }

            HighGui.imshow("img2", frame);
            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q'|| keyboard == 27) {
                break;
            }
        }
        System.exit(0);
    }
}

public class CamshiftDemo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new Camshift().run(args);
    }
}
