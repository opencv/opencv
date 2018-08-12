import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

class ChangingContrastBrightnessImage {
    private static int MAX_VALUE_ALPHA = 500;
    private static int MAX_VALUE_BETA_GAMMA = 200;
    private static final String WINDOW_NAME = "Changing the contrast and brightness of an image demo";
    private static final String ALPHA_NAME = "Alpha gain (contrast)";
    private static final String BETA_NAME = "Beta bias (brightness)";
    private static final String GAMMA_NAME = "Gamma correction";
    private JFrame frame;
    private Mat matImgSrc = new Mat();
    private JLabel imgSrcLabel;
    private JLabel imgModifLabel;
    private JPanel controlPanel;
    private JPanel alphaBetaPanel;
    private JPanel gammaPanel;
    private double alphaValue = 1.0;
    private double betaValue = 0.0;
    private double gammaValue = 1.0;
    private JCheckBox methodCheckBox;
    private JSlider sliderAlpha;
    private JSlider sliderBeta;
    private JSlider sliderGamma;

    public ChangingContrastBrightnessImage(String[] args) {
        String imagePath = args.length > 0 ? args[0] : "../data/lena.jpg";
        matImgSrc = Imgcodecs.imread(imagePath);
        if (matImgSrc.empty()) {
            System.out.println("Empty image: " + imagePath);
            System.exit(0);
        }

        // Create and set up the window.
        frame = new JFrame(WINDOW_NAME);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(matImgSrc);
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

        controlPanel = new JPanel();
        controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.PAGE_AXIS));

        methodCheckBox = new JCheckBox("Do gamma correction");
        methodCheckBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JCheckBox cb = (JCheckBox) e.getSource();
                if (cb.isSelected()) {
                    controlPanel.remove(alphaBetaPanel);
                    controlPanel.add(gammaPanel);
                    performGammaCorrection();
                    frame.revalidate();
                    frame.repaint();
                    frame.pack();
                } else {
                    controlPanel.remove(gammaPanel);
                    controlPanel.add(alphaBetaPanel);
                    performLinearTransformation();
                    frame.revalidate();
                    frame.repaint();
                    frame.pack();
                }
            }
        });
        controlPanel.add(methodCheckBox);

        alphaBetaPanel = new JPanel();
        alphaBetaPanel.setLayout(new BoxLayout(alphaBetaPanel, BoxLayout.PAGE_AXIS));
        alphaBetaPanel.add(new JLabel(ALPHA_NAME));
        sliderAlpha = new JSlider(0, MAX_VALUE_ALPHA, 100);
        sliderAlpha.setMajorTickSpacing(50);
        sliderAlpha.setMinorTickSpacing(10);
        sliderAlpha.setPaintTicks(true);
        sliderAlpha.setPaintLabels(true);
        sliderAlpha.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                alphaValue = sliderAlpha.getValue() / 100.0;
                performLinearTransformation();
            }
        });
        alphaBetaPanel.add(sliderAlpha);

        alphaBetaPanel.add(new JLabel(BETA_NAME));
        sliderBeta = new JSlider(0, MAX_VALUE_BETA_GAMMA, 100);
        sliderBeta.setMajorTickSpacing(20);
        sliderBeta.setMinorTickSpacing(5);
        sliderBeta.setPaintTicks(true);
        sliderBeta.setPaintLabels(true);
        sliderBeta.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                betaValue = sliderBeta.getValue() - 100;
                performLinearTransformation();
            }
        });
        alphaBetaPanel.add(sliderBeta);
        controlPanel.add(alphaBetaPanel);

        gammaPanel = new JPanel();
        gammaPanel.setLayout(new BoxLayout(gammaPanel, BoxLayout.PAGE_AXIS));
        gammaPanel.add(new JLabel(GAMMA_NAME));
        sliderGamma = new JSlider(0, MAX_VALUE_BETA_GAMMA, 100);
        sliderGamma.setMajorTickSpacing(20);
        sliderGamma.setMinorTickSpacing(5);
        sliderGamma.setPaintTicks(true);
        sliderGamma.setPaintLabels(true);
        sliderGamma.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                gammaValue = sliderGamma.getValue() / 100.0;
                performGammaCorrection();
            }
        });
        gammaPanel.add(sliderGamma);

        pane.add(controlPanel, BorderLayout.PAGE_START);
        JPanel framePanel = new JPanel();
        imgSrcLabel = new JLabel(new ImageIcon(img));
        framePanel.add(imgSrcLabel);
        imgModifLabel = new JLabel(new ImageIcon(img));
        framePanel.add(imgModifLabel);
        pane.add(framePanel, BorderLayout.CENTER);
    }

    private void performLinearTransformation() {
        Mat img = new Mat();
        matImgSrc.convertTo(img, -1, alphaValue, betaValue);
        imgModifLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(img)));
        frame.repaint();
    }

    private byte saturate(double val) {
        int iVal = (int) Math.round(val);
        iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
        return (byte) iVal;
    }

    private void performGammaCorrection() {
        //! [changing-contrast-brightness-gamma-correction]
        Mat lookUpTable = new Mat(1, 256, CvType.CV_8U);
        byte[] lookUpTableData = new byte[(int) (lookUpTable.total()*lookUpTable.channels())];
        for (int i = 0; i < lookUpTable.cols(); i++) {
            lookUpTableData[i] = saturate(Math.pow(i / 255.0, gammaValue) * 255.0);
        }
        lookUpTable.put(0, 0, lookUpTableData);
        Mat img = new Mat();
        Core.LUT(matImgSrc, lookUpTable, img);
        //! [changing-contrast-brightness-gamma-correction]

        imgModifLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(img)));
        frame.repaint();
    }
}

public class ChangingContrastBrightnessImageDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ChangingContrastBrightnessImage(args);
            }
        });
    }
}
