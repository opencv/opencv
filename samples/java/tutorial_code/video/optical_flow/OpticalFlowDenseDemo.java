import java.util.ArrayList;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;


class OptFlowDense {
    public void run(String[] args) {
        String filename = args[0];
        VideoCapture capture = new VideoCapture(filename);
        if (!capture.isOpened()) {
            //error in opening the video input
            System.out.println("Unable to open file!");
            System.exit(-1);
        }

        Mat frame1 = new Mat() , prvs = new Mat();
        capture.read(frame1);
        Imgproc.cvtColor(frame1, prvs, Imgproc.COLOR_BGR2GRAY);

        while (true) {
            Mat frame2 = new Mat(), next = new Mat();
            capture.read(frame2);
            if (frame2.empty()) {
                break;
            }
            Imgproc.cvtColor(frame2, next, Imgproc.COLOR_BGR2GRAY);

            Mat flow = new Mat(prvs.size(), CvType.CV_32FC2);
            Video.calcOpticalFlowFarneback(prvs, next, flow,0.5,3,15,3,5,1.2,0);

            // visualization
            ArrayList<Mat> flow_parts = new ArrayList<>(2);
            Core.split(flow, flow_parts);
            Mat magnitude = new Mat(), angle = new Mat(), magn_norm = new Mat();
            Core.cartToPolar(flow_parts.get(0), flow_parts.get(1), magnitude, angle,true);
            Core.normalize(magnitude, magn_norm,0.0,1.0, Core.NORM_MINMAX);
            float factor = (float) ((1.0/360.0)*(180.0/255.0));
            Mat new_angle = new Mat();
            Core.multiply(angle, new Scalar(factor), new_angle);

            //build hsv image
            ArrayList<Mat> _hsv = new ArrayList<>() ;
            Mat hsv = new Mat(), hsv8 = new Mat(), bgr = new Mat();

            _hsv.add(new_angle);
            _hsv.add(Mat.ones(angle.size(), CvType.CV_32F));
            _hsv.add(magn_norm);
            Core.merge(_hsv, hsv);
            hsv.convertTo(hsv8, CvType.CV_8U, 255.0);
            Imgproc.cvtColor(hsv8, bgr, Imgproc.COLOR_HSV2BGR);

            HighGui.imshow("frame2", bgr);

            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
            prvs = next;
        }
        System.exit(0);
    }
}

public class OpticalFlowDenseDemo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new OptFlowDense().run(args);
    }
}
