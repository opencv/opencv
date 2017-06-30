import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

class MatMaskOperationsRun {

    public void run(String[] args) {

        String filename = "../data/lena.jpg";

        int img_codec = Imgcodecs.IMREAD_COLOR;
        if (args.length != 0) {
            filename = args[0];
            if (args.length >= 2 && args[1].equals("G"))
                img_codec = Imgcodecs.IMREAD_GRAYSCALE;
        }

        Mat src = Imgcodecs.imread(filename, img_codec);

        if (src.empty()) {
            System.out.println("Can't open image [" + filename + "]");
            System.out.println("Program Arguments: [image_path -- default ../data/lena.jpg] [G -- grayscale]");
            System.exit(-1);
        }

        Image img = toBufferedImage(src);
        displayImage("Input", img, 0, 200);
        double t = System.currentTimeMillis();

        Mat dst0 = sharpen(src, new Mat());

        t = ((double) System.currentTimeMillis() - t) / 1000;
        System.out.println("Hand written function time passed in seconds: " + t);

        Image img2 = toBufferedImage(dst0);
        displayImage("Output", img2, 400, 400);

        //![kern]
        Mat kern = new Mat(3, 3, CvType.CV_8S);
        int row = 0, col = 0;
        kern.put(row, col, 0, -1, 0, -1, 5, -1, 0, -1, 0);
        //![kern]

        t = System.currentTimeMillis();

        Mat dst1 = new Mat();
        //![filter2D]
        Imgproc.filter2D(src, dst1, src.depth(), kern);
        //![filter2D]
        t = ((double) System.currentTimeMillis() - t) / 1000;
        System.out.println("Built-in filter2D time passed in seconds:     " + t);

        Image img3 = toBufferedImage(dst1);
        displayImage("Output", img3, 800, 400);
    }

    //! [basic_method]
    public static double saturate(double x) {
        return x > 255.0 ? 255.0 : (x < 0.0 ? 0.0 : x);
    }

    public Mat sharpen(Mat myImage, Mat Result) {
        //! [8_bit]
        myImage.convertTo(myImage, CvType.CV_8U);
        //! [8_bit]

        //! [create_channels]
        int nChannels = myImage.channels();
        Result.create(myImage.size(), myImage.type());
        //! [create_channels]

        //! [basic_method_loop]
        for (int j = 1; j < myImage.rows() - 1; ++j) {
            for (int i = 1; i < myImage.cols() - 1; ++i) {
                double sum[] = new double[nChannels];

                for (int k = 0; k < nChannels; ++k) {

                    double top = -myImage.get(j - 1, i)[k];
                    double bottom = -myImage.get(j + 1, i)[k];
                    double center = (5 * myImage.get(j, i)[k]);
                    double left = -myImage.get(j, i - 1)[k];
                    double right = -myImage.get(j, i + 1)[k];

                    sum[k] = saturate(top + bottom + center + left + right);
                }

                Result.put(j, i, sum);
            }
        }
        //! [basic_method_loop]

        //! [borders]
        Result.row(0).setTo(new Scalar(0));
        Result.row(Result.rows() - 1).setTo(new Scalar(0));
        Result.col(0).setTo(new Scalar(0));
        Result.col(Result.cols() - 1).setTo(new Scalar(0));
        //! [borders]

        return Result;
    }
    //! [basic_method]

    public Image toBufferedImage(Mat m) {
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

    public void displayImage(String title, Image img, int x, int y) {
        ImageIcon icon = new ImageIcon(img);
        JFrame frame = new JFrame(title);
        JLabel lbl = new JLabel(icon);
        frame.add(lbl);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setLocation(x, y);
        frame.setVisible(true);
    }
}

public class MatMaskOperations {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new MatMaskOperationsRun().run(args); // run code
    }
}
