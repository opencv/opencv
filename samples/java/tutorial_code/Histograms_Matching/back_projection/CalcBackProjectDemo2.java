import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Image;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Arrays;
import java.util.List;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class CalcBackProject2 {
    private Mat src;
    private Mat hsv = new Mat();
    private Mat mask = new Mat();
    private JFrame frame;
    private JLabel imgLabel;
    private JLabel backprojLabel;
    private JLabel maskImgLabel;
    private static final int MAX_SLIDER = 255;
    private int low = 20;
    private int up = 20;

    public CalcBackProject2(String[] args) {
        /// Read the image
        if (args.length != 1) {
            System.err.println("You must supply one argument that corresponds to the path to the image.");
            System.exit(0);
        }

        src = Imgcodecs.imread(args[0]);
        if (src.empty()) {
            System.err.println("Empty image: " + args[0]);
            System.exit(0);
        }

        /// Transform it to HSV
        Imgproc.cvtColor(src, hsv, Imgproc.COLOR_BGR2HSV);

        // Create and set up the window.
        frame = new JFrame("Back Projection 2 demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(src);
        addComponentsToPane(frame.getContentPane(), img);
        // Use the content pane's default BorderLayout. No need for
        // setLayout(new BorderLayout());
        // Display the window.
        frame.pack();
        frame.setVisible(true);
    }

    private void addComponentsToPane(Container pane, Image img) {
        if (!(pane.getLayout() instanceof BorderLayout)) {
            pane.add(new JLabel("Container doesn't use BorderLayout!"));
            return;
        }

        /// Set Trackbars for floodfill thresholds
        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));

        sliderPanel.add(new JLabel("Low thresh"));
        JSlider slider = new JSlider(0, MAX_SLIDER, low);
        slider.setMajorTickSpacing(20);
        slider.setMinorTickSpacing(10);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                low = source.getValue();
            }
        });
        sliderPanel.add(slider);
        pane.add(sliderPanel, BorderLayout.PAGE_START);

        sliderPanel.add(new JLabel("High thresh"));
        slider = new JSlider(0, MAX_SLIDER, up);
        slider.setMajorTickSpacing(20);
        slider.setMinorTickSpacing(10);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                up = source.getValue();
            }
        });
        sliderPanel.add(slider);
        pane.add(sliderPanel, BorderLayout.PAGE_START);

        JPanel imgPanel = new JPanel();
        imgLabel = new JLabel(new ImageIcon(img));
        /// Set a Mouse Callback
        imgLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                update(e.getX(), e.getY());
            }
        });
        imgPanel.add(imgLabel);

        maskImgLabel = new JLabel();
        imgPanel.add(maskImgLabel);

        backprojLabel = new JLabel();
        imgPanel.add(backprojLabel);

        pane.add(imgPanel, BorderLayout.CENTER);
    }

    private void update(int x, int y) {
        // Fill and get the mask
        Point seed = new Point(x, y);

        int newMaskVal = 255;
        Scalar newVal = new Scalar(120, 120, 120);

        int connectivity = 8;
        int flags = connectivity + (newMaskVal << 8) + Imgproc.FLOODFILL_FIXED_RANGE + Imgproc.FLOODFILL_MASK_ONLY;

        Mat mask2 = Mat.zeros(src.rows() + 2, src.cols() + 2, CvType.CV_8U);
        Imgproc.floodFill(src, mask2, seed, newVal, new Rect(), new Scalar(low, low, low), new Scalar(up, up, up), flags);
        mask = mask2.submat(new Range(1, mask2.rows() - 1), new Range(1, mask2.cols() - 1));

        Image maskImg = HighGui.toBufferedImage(mask);
        maskImgLabel.setIcon(new ImageIcon(maskImg));

        int hBins = 30, sBins = 32;
        int[] histSize = { hBins, sBins };
        float[] ranges = { 0, 180, 0, 256 };
        int[] channels = { 0, 1 };

        /// Get the Histogram and normalize it
        Mat hist = new Mat();
        List<Mat> hsvList = Arrays.asList(hsv);
        Imgproc.calcHist(hsvList, new MatOfInt(channels), mask, hist, new MatOfInt(histSize), new MatOfFloat(ranges), false );

        Core.normalize(hist, hist, 0, 255, Core.NORM_MINMAX);

        /// Get Backprojection
        Mat backproj = new Mat();
        Imgproc.calcBackProject(hsvList, new MatOfInt(channels), hist, backproj, new MatOfFloat(ranges), 1);

        Image backprojImg = HighGui.toBufferedImage(backproj);
        backprojLabel.setIcon(new ImageIcon(backprojImg));

        frame.repaint();
        frame.pack();
    }
}

public class CalcBackProjectDemo2 {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new CalcBackProject2(args);
            }
        });
    }
}
