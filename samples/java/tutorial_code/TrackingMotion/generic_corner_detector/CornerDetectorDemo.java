import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Image;
import java.util.Random;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class CornerDetector {
    private Mat src = new Mat();
    private Mat srcGray = new Mat();
    private Mat harrisDst = new Mat();
    private Mat shiTomasiDst = new Mat();
    private Mat harrisCopy = new Mat();
    private Mat shiTomasiCopy = new Mat();
    private Mat Mc = new Mat();
    private JFrame frame;
    private JLabel harrisImgLabel;
    private JLabel shiTomasiImgLabel;
    private static final int MAX_QUALITY_LEVEL = 100;
    private int qualityLevel = 50;
    private double harrisMinVal;
    private double harrisMaxVal;
    private double shiTomasiMinVal;
    private double shiTomasiMaxVal;
    private Random rng = new Random(12345);

    public CornerDetector(String[] args) {
        /// Load source image and convert it to gray
        String filename = args.length > 0 ? args[0] : "../data/building.jpg";
        src = Imgcodecs.imread(filename);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }

        Imgproc.cvtColor(src, srcGray, Imgproc.COLOR_BGR2GRAY);

        // Create and set up the window.
        frame = new JFrame("Creating your own corner detector demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(src);
        addComponentsToPane(frame.getContentPane(), img);
        // Use the content pane's default BorderLayout. No need for
        // setLayout(new BorderLayout());
        // Display the window.
        frame.pack();
        frame.setVisible(true);

        /// Set some parameters
        int blockSize = 3, apertureSize = 3;

        /// My Harris matrix -- Using cornerEigenValsAndVecs
        Imgproc.cornerEigenValsAndVecs(srcGray, harrisDst, blockSize, apertureSize);

        /* calculate Mc */
        Mc = Mat.zeros(srcGray.size(), CvType.CV_32F);

        float[] harrisData = new float[(int) (harrisDst.total() * harrisDst.channels())];
        harrisDst.get(0, 0, harrisData);
        float[] McData = new float[(int) (Mc.total() * Mc.channels())];
        Mc.get(0, 0, McData);

        for( int i = 0; i < srcGray.rows(); i++ ) {
            for( int j = 0; j < srcGray.cols(); j++ ) {
                float lambda1 = harrisData[(i*srcGray.cols() + j) * 6];
                float lambda2 = harrisData[(i*srcGray.cols() + j) * 6 + 1];
                McData[i*srcGray.cols()+j] = (float) (lambda1*lambda2 - 0.04f*Math.pow( ( lambda1 + lambda2 ), 2 ));
            }
        }
        Mc.put(0, 0, McData);

        MinMaxLocResult res = Core.minMaxLoc(Mc);
        harrisMinVal = res.minVal;
        harrisMaxVal = res.maxVal;

        /// My Shi-Tomasi -- Using cornerMinEigenVal
        Imgproc.cornerMinEigenVal(srcGray, shiTomasiDst, blockSize, apertureSize);
        res = Core.minMaxLoc(shiTomasiDst);
        shiTomasiMinVal = res.minVal;
        shiTomasiMaxVal = res.maxVal;

        update();
    }

    private void addComponentsToPane(Container pane, Image img) {
        if (!(pane.getLayout() instanceof BorderLayout)) {
            pane.add(new JLabel("Container doesn't use BorderLayout!"));
            return;
        }

        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));

        sliderPanel.add(new JLabel("Max  corners:"));
        JSlider slider = new JSlider(0, MAX_QUALITY_LEVEL, qualityLevel);
        slider.setMajorTickSpacing(20);
        slider.setMinorTickSpacing(10);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                qualityLevel = source.getValue();
                update();
            }
        });
        sliderPanel.add(slider);
        pane.add(sliderPanel, BorderLayout.PAGE_START);

        JPanel imgPanel = new JPanel();
        harrisImgLabel = new JLabel(new ImageIcon(img));
        shiTomasiImgLabel = new JLabel(new ImageIcon(img));
        imgPanel.add(harrisImgLabel);
        imgPanel.add(shiTomasiImgLabel);
        pane.add(imgPanel, BorderLayout.CENTER);
    }

    private void update() {
        int qualityLevelVal = Math.max(qualityLevel, 1);

        //Harris
        harrisCopy = src.clone();

        float[] McData = new float[(int) (Mc.total() * Mc.channels())];
        Mc.get(0, 0, McData);
        for (int i = 0; i < srcGray.rows(); i++) {
            for (int j = 0; j < srcGray.cols(); j++) {
                if (McData[i * srcGray.cols() + j] > harrisMinVal
                        + (harrisMaxVal - harrisMinVal) * qualityLevelVal / MAX_QUALITY_LEVEL) {
                    Imgproc.circle(harrisCopy, new Point(j, i), 4,
                            new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)), Core.FILLED);
                }
            }
        }

        //Shi-Tomasi
        shiTomasiCopy = src.clone();

        float[] shiTomasiData = new float[(int) (shiTomasiDst.total() * shiTomasiDst.channels())];
        shiTomasiDst.get(0, 0, shiTomasiData);
        for (int i = 0; i < srcGray.rows(); i++) {
            for (int j = 0; j < srcGray.cols(); j++) {
                if (shiTomasiData[i * srcGray.cols() + j] > shiTomasiMinVal
                        + (shiTomasiMaxVal - shiTomasiMinVal) * qualityLevelVal / MAX_QUALITY_LEVEL) {
                    Imgproc.circle(shiTomasiCopy, new Point(j, i), 4,
                            new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)), Core.FILLED);
                }
            }
        }

        harrisImgLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(harrisCopy)));
        shiTomasiImgLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(shiTomasiCopy)));
        frame.repaint();
    }
}

public class CornerDetectorDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new CornerDetector(args);
            }
        });
    }
}
