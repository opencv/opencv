import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Image;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.List;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingWorker;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class ThresholdInRange {
    private static int MAX_VALUE = 255;
    private static int MAX_VALUE_H = 360/2;
    private static final String WINDOW_NAME = "Thresholding Operations using inRange demo";
    private static final String LOW_H_NAME = "Low H";
    private static final String LOW_S_NAME = "Low S";
    private static final String LOW_V_NAME = "Low V";
    private static final String HIGH_H_NAME = "High H";
    private static final String HIGH_S_NAME = "High S";
    private static final String HIGH_V_NAME = "High V";
    private JSlider sliderLowH;
    private JSlider sliderHighH;
    private JSlider sliderLowS;
    private JSlider sliderHighS;
    private JSlider sliderLowV;
    private JSlider sliderHighV;
    private VideoCapture cap;
    private Mat matFrame = new Mat();
    private JFrame frame;
    private JLabel imgCaptureLabel;
    private JLabel imgDetectionLabel;
    private CaptureTask captureTask;

    public ThresholdInRange(String[] args) {
        int cameraDevice = 0;
        if (args.length > 0) {
            cameraDevice = Integer.parseInt(args[0]);
        }
        //! [cap]
        cap = new VideoCapture(cameraDevice);
        //! [cap]
        if (!cap.isOpened()) {
            System.err.println("Cannot open camera: " + cameraDevice);
            System.exit(0);
        }
        if (!cap.read(matFrame)) {
            System.err.println("Cannot read camera stream.");
            System.exit(0);
        }

        //! [window]
        // Create and set up the window.
        frame = new JFrame(WINDOW_NAME);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent windowEvent) {
                captureTask.cancel(true);
            }
        });
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(matFrame);
        addComponentsToPane(frame.getContentPane(), img);
        // Use the content pane's default BorderLayout. No need for
        // setLayout(new BorderLayout());
        // Display the window.
        frame.pack();
        frame.setVisible(true);
        //! [window]

        captureTask = new CaptureTask();
        captureTask.execute();
    }

    //! [while]
    private class CaptureTask extends SwingWorker<Void, Mat> {
        @Override
        protected Void doInBackground() {
            Mat matFrame = new Mat();

            while (!isCancelled()) {
                if (!cap.read(matFrame)) {
                    break;
                }
                publish(matFrame.clone());
            }
            return null;
        }

        @Override
        protected void process(List<Mat> frames) {
            Mat frame = frames.get(frames.size() - 1);
            Mat frameHSV = new Mat();
            Imgproc.cvtColor(frame, frameHSV, Imgproc.COLOR_BGR2HSV);
            Mat thresh = new Mat();
            Core.inRange(frameHSV, new Scalar(sliderLowH.getValue(), sliderLowS.getValue(), sliderLowV.getValue()),
                    new Scalar(sliderHighH.getValue(), sliderHighS.getValue(), sliderHighV.getValue()), thresh);
            update(frame, thresh);
        }
    }
    //! [while]

    private void addComponentsToPane(Container pane, Image img) {
        if (!(pane.getLayout() instanceof BorderLayout)) {
            pane.add(new JLabel("Container doesn't use BorderLayout!"));
            return;
        }

        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));

        //! [trackbar]
        sliderPanel.add(new JLabel(LOW_H_NAME));
        sliderLowH = new JSlider(0, MAX_VALUE_H, 0);
        sliderLowH.setMajorTickSpacing(50);
        sliderLowH.setMinorTickSpacing(10);
        sliderLowH.setPaintTicks(true);
        sliderLowH.setPaintLabels(true);
        sliderPanel.add(sliderLowH);

        sliderPanel.add(new JLabel(HIGH_H_NAME));
        sliderHighH = new JSlider(0, MAX_VALUE_H, MAX_VALUE_H);
        sliderHighH.setMajorTickSpacing(50);
        sliderHighH.setMinorTickSpacing(10);
        sliderHighH.setPaintTicks(true);
        sliderHighH.setPaintLabels(true);
        sliderPanel.add(sliderHighH);

        sliderPanel.add(new JLabel(LOW_S_NAME));
        sliderLowS = new JSlider(0, MAX_VALUE, 0);
        sliderLowS.setMajorTickSpacing(50);
        sliderLowS.setMinorTickSpacing(10);
        sliderLowS.setPaintTicks(true);
        sliderLowS.setPaintLabels(true);
        sliderPanel.add(sliderLowS);

        sliderPanel.add(new JLabel(HIGH_S_NAME));
        sliderHighS = new JSlider(0, MAX_VALUE, MAX_VALUE);
        sliderHighS.setMajorTickSpacing(50);
        sliderHighS.setMinorTickSpacing(10);
        sliderHighS.setPaintTicks(true);
        sliderHighS.setPaintLabels(true);
        sliderPanel.add(sliderHighS);

        sliderPanel.add(new JLabel(LOW_V_NAME));
        sliderLowV = new JSlider(0, MAX_VALUE, 0);
        sliderLowV.setMajorTickSpacing(50);
        sliderLowV.setMinorTickSpacing(10);
        sliderLowV.setPaintTicks(true);
        sliderLowV.setPaintLabels(true);
        sliderPanel.add(sliderLowV);

        sliderPanel.add(new JLabel(HIGH_V_NAME));
        sliderHighV = new JSlider(0, MAX_VALUE, MAX_VALUE);
        sliderHighV.setMajorTickSpacing(50);
        sliderHighV.setMinorTickSpacing(10);
        sliderHighV.setPaintTicks(true);
        sliderHighV.setPaintLabels(true);
        sliderPanel.add(sliderHighV);
        //! [trackbar]

        //! [low]
        sliderLowH.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                int valH = Math.min(sliderHighH.getValue()-1, source.getValue());
                sliderLowH.setValue(valH);
            }
        });
        //! [low]
        //! [high]
        sliderHighH.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                int valH = Math.max(source.getValue(), sliderLowH.getValue()+1);
                sliderHighH.setValue(valH);
            }
        });
        //! [high]
        sliderLowS.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                int valS = Math.min(sliderHighS.getValue()-1, source.getValue());
                sliderLowS.setValue(valS);
            }
        });
        sliderHighS.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                int valS = Math.max(source.getValue(), sliderLowS.getValue()+1);
                sliderHighS.setValue(valS);
            }
        });
        sliderLowV.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                int valV = Math.min(sliderHighV.getValue()-1, source.getValue());
                sliderLowV.setValue(valV);
            }
        });
        sliderHighV.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                int valV = Math.max(source.getValue(), sliderLowV.getValue()+1);
                sliderHighV.setValue(valV);
            }
        });

        pane.add(sliderPanel, BorderLayout.PAGE_START);
        JPanel framePanel = new JPanel();
        imgCaptureLabel = new JLabel(new ImageIcon(img));
        framePanel.add(imgCaptureLabel);
        imgDetectionLabel = new JLabel(new ImageIcon(img));
        framePanel.add(imgDetectionLabel);
        pane.add(framePanel, BorderLayout.CENTER);
    }

    private void update(Mat imgCapture, Mat imgThresh) {
        //! [show]
        imgCaptureLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(imgCapture)));
        imgDetectionLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(imgThresh)));
        frame.repaint();
        //! [show]
    }

    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ThresholdInRange(args);
            }
        });
    }
}
