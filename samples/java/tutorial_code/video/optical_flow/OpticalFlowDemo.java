import java.util.ArrayList;
import java.util.Random;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

class OptFlow {
    public void run(String[] args) {
        String filename = args[0];
        VideoCapture capture = new VideoCapture(filename);
        if (!capture.isOpened()) {
            System.out.println("Unable to open this file");
            System.exit(-1);
        }


        // Create some random colors
        Scalar[] colors = new Scalar[100];
        Random rng = new Random();
        for (int i = 0 ; i < 100 ; i++) {
            int r = rng.nextInt(256);
            int g = rng.nextInt(256);
            int b = rng.nextInt(256);
            colors[i] = new Scalar(r, g, b);
        }

        Mat old_frame = new Mat() , old_gray = new Mat();

        // Since the function Imgproc.goodFeaturesToTrack requires MatofPoint
        // therefore first p0MatofPoint is passed to the function and then converted to MatOfPoint2f
        MatOfPoint p0MatofPoint = new MatOfPoint();
        capture.read(old_frame);
        Imgproc.cvtColor(old_frame, old_gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.goodFeaturesToTrack(old_gray, p0MatofPoint,100,0.3,7, new Mat(),7,false,0.04);

        MatOfPoint2f p0 = new MatOfPoint2f(p0MatofPoint.toArray()) , p1 = new MatOfPoint2f();

        // Create a mask image for drawing purposes
        Mat mask = Mat.zeros(old_frame.size(), old_frame.type());

        while (true) {
            Mat frame = new Mat(), frame_gray = new Mat();
            capture.read(frame);
            if (frame.empty()) {
                break;
            }

            Imgproc.cvtColor(frame, frame_gray, Imgproc.COLOR_BGR2GRAY);

            // calculate optical flow
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,10,0.03);
            Video.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, new Size(15,15),2, criteria);

            byte StatusArr[] = status.toArray();
            Point p0Arr[] = p0.toArray();
            Point p1Arr[] = p1.toArray();
            ArrayList<Point> good_new = new ArrayList<>();

            for (int i = 0; i<StatusArr.length ; i++ ) {
                if (StatusArr[i] == 1) {
                    good_new.add(p1Arr[i]);
                    Imgproc.line(mask, p1Arr[i], p0Arr[i], colors[i],2);
                    Imgproc.circle(frame, p1Arr[i],5, colors[i],-1);
                }
            }

            Mat img = new Mat();
            Core.add(frame, mask, img);

            HighGui.imshow("Frame", img);

            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }

            // Now update the previous frame and previous points
            old_gray = frame_gray.clone();
            Point[] good_new_arr = new Point[good_new.size()];
            good_new_arr = good_new.toArray(good_new_arr);
            p0 = new MatOfPoint2f(good_new_arr);
        }
        System.exit(0);
    }
}

public class OpticalFlowDemo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new OptFlow().run(args);
    }
}
