import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

class BackgroundSubtraction {
    public void run(String[] args) {
        String input = args.length > 0 ? args[0] : "../data/vtest.avi";
        boolean useMOG2 = args.length > 1 ? args[1] == "MOG2" : true;

        //! [create]
        BackgroundSubtractor backSub;
        if (useMOG2) {
            backSub = Video.createBackgroundSubtractorMOG2();
        } else {
            backSub = Video.createBackgroundSubtractorKNN();
        }
        //! [create]

        //! [capture]
        VideoCapture capture = new VideoCapture(input);
        if (!capture.isOpened()) {
            System.err.println("Unable to open: " + input);
            System.exit(0);
        }
        //! [capture]

        Mat frame = new Mat(), fgMask = new Mat();
        while (true) {
            capture.read(frame);
            if (frame.empty()) {
                break;
            }

            //! [apply]
            // update the background model
            backSub.apply(frame, fgMask);
            //! [apply]

            //! [display_frame_number]
            // get the frame number and write it on the current frame
            Imgproc.rectangle(frame, new Point(10, 2), new Point(100, 20), new Scalar(255, 255, 255), -1);
            String frameNumberString = String.format("%d", (int)capture.get(Videoio.CAP_PROP_POS_FRAMES));
            Imgproc.putText(frame, frameNumberString, new Point(15, 15), Core.FONT_HERSHEY_SIMPLEX, 0.5,
                    new Scalar(0, 0, 0));
            //! [display_frame_number]

            //! [show]
            // show the current frame and the fg masks
            HighGui.imshow("Frame", frame);
            HighGui.imshow("FG Mask", fgMask);
            //! [show]

            // get the input from the keyboard
            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
        }

        HighGui.waitKey();
        System.exit(0);
    }
}

public class BackgroundSubtractionDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new BackgroundSubtraction().run(args);
    }
}
