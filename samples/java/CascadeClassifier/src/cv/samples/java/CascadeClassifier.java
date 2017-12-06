package cv.samples.java;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.circle;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.ellipse;
import static org.opencv.imgproc.Imgproc.equalizeHist;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStreamReader;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.videoio.VideoCapture;

public class CascadeClassifier {

    static {
        System.setProperty("java.library.path", System.getProperty("java.library.path") + ":"
                + "/home/styagi/.workspace/Downloads/opencv-3.2.0/build/lib");
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    private static org.opencv.objdetect.CascadeClassifier face_cascade;
    private static org.opencv.objdetect.CascadeClassifier eyes_cascade;
    private static ImageIcon icon;
    private static JFrame jframe;

    static final String window_name = "Capture - Face detection";
    private static final int X = 800;
    private static final int Y = 600;

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("{help h||}");
            System.out.println("{face_cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}");
            System.out.println("{eyes_cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}");
        }

        System.out.println(
                "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
                        + "You can use Haar or LBP features.\n\n");

        String face_cascade_name = args[0];
        String eyes_cascade_name = args[1];
        VideoCapture capture = new VideoCapture(0);

        // -- 1. Load the cascades
        try {
            face_cascade = new org.opencv.objdetect.CascadeClassifier(face_cascade_name);
            eyes_cascade = new org.opencv.objdetect.CascadeClassifier(eyes_cascade_name);
        } catch (Exception e) {
            e.printStackTrace(System.err);
            System.exit(-1);
        }

        // -- 2. Read the video stream
        capture.open(0);
        if (!capture.isOpened()) {
            System.out.println("--(!)Error opening video capture\n");
            System.exit(-1);
        }

        try (InputStreamReader reader = new InputStreamReader(System.in)) {
            icon = new ImageIcon();
            jframe = new JFrame(window_name);
            JLabel lbl = new JLabel(icon);
            jframe.add(lbl);
            jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            boolean set = false;

            Mat frame = new Mat();
            while (capture.read(frame)) {
                if (frame.empty()) {
                    System.out.println(" --(!) No captured frame -- Break!");
                    break;
                }
                // -- 3. Apply the classifier to the frame
                detectAndDisplay(frame);
                if (!set) {
                    set = true;
                    jframe.pack();
                    jframe.setLocation(X, Y);
                    jframe.setVisible(true);
                }
                frame = new Mat();
            }

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        System.exit(0);
    }

    /** @function detectAndDisplay */
    private static void detectAndDisplay(Mat frame) {
        MatOfRect facesm = new MatOfRect();
        Mat frame_gray = new Mat();

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        // -- Detect faces
        face_cascade.detectMultiScale(frame_gray, facesm);

        Rect[] faces = facesm.toArray();

        /*
         * if (faces.length > 0) {
         * System.out.println(String.format("Found %s face(s)", faces.length));
         * }
         */

        for (int i = 0; i < faces.length; i++) {
            Point center = new Point(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
            ellipse(frame, center, new Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360,
                    new Scalar(255, 0, 255), 4, 8, 0);

            Mat faceROI = frame_gray.submat(faces[i]);
            MatOfRect eyesm = new MatOfRect();

            // -- In each face, detect eyes
            eyes_cascade.detectMultiScale(faceROI, eyesm);

            Rect[] eyes = eyesm.toArray();
            for (int j = 0; j < eyes.length; j++) {
                Point eye_center = new Point(faces[i].x + eyes[j].x + eyes[j].width / 2,
                        faces[i].y + eyes[j].y + eyes[j].height / 2);
                long radius = Math.round((eyes[j].width + eyes[j].height) * 0.25);
                circle(frame, eye_center, (int) radius, new Scalar(255, 0, 0), 4, 8, 0);
            }
        }
        // -- Show what you got
        imshow(window_name, frame);
    }

    private static void imshow(String windowName, Mat frame) {
        displayImage(windowName, toBufferedImage(frame), X, Y);
    }

    public static Image toBufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels() * m.cols() * m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    public static void displayImage(String title, Image img, int x, int y) {
        icon.setImage(img);
        jframe.revalidate();
        jframe.repaint();
    }
}
