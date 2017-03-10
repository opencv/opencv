import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

class MatMaskOperationsRun {

    public void run() {

    //! [laod_image]
        Mat I = Imgcodecs.imread("../data/lena.jpg");
        if(I.empty())
            System.out.println("Error opening image");
    //! [laod_image]

        Image img = toBufferedImage( I );
        displayImage("Input Image" , img, 0, 200 );

        double t = System.currentTimeMillis();

        Mat J = sharpen(I, new Mat());

        t = ((double)System.currentTimeMillis() - t)/1000;
        System.out.println("Hand written function times passed in seconds: " + t);

        Image img2 = toBufferedImage( J );
        displayImage("Output Image" , img2, 400, 400 );

        Mat K = new Mat();
    //![kern]
        Mat kern = new Mat( 3, 3, CvType.CV_8S );
        int row = 0, col = 0;
        kern.put(row ,col, 0, -1, 0, -1, 5, -1, 0, -1, 0 );
    //![kern]

        System.out.println("kern = \n" + kern.dump());

        t = System.currentTimeMillis();

    //![filter2D]
        Imgproc.filter2D(I, K, I.depth(), kern );
    //![filter2D]

        t = ((double)System.currentTimeMillis() - t)/1000;
        System.out.println("Built-in filter2D time passed in seconds:      " + t);

        Image img3 = toBufferedImage( J );
        displayImage("filter2D Output Image" , img3, 800, 400 );
    }

    //! [basic_method]
    public static double saturateCastUchar(double x) {
        return x > 255.0 ? 255.0 : (x < 0.0 ? 0.0 : x);
    }

    public Mat sharpen(Mat myImage, Mat Result)
    {
      //! [8_bit]
        myImage.convertTo(myImage, CvType.CV_8U);
      //! [8_bit]

      //! [create_channels]
        int nChannels = myImage.channels();
        Result.create(myImage.size(),myImage.type());
      //! [create_channels]

      //! [basic_method_loop]
        for(int j = 1 ; j < myImage.rows()-1; ++j)
        {
            for(int i = 1 ; i < myImage.cols()-1; ++i)
            {
                double sum[] = new double[nChannels];

                for(int k = 0; k < nChannels; ++k) {

                    double top = -myImage.get(j - 1, i)[k];
                    double bottom = -myImage.get(j + 1, i)[k];
                    double center = (5 * myImage.get(j, i)[k]);
                    double left = -myImage.get(j , i - 1)[k];
                    double right = -myImage.get(j , i + 1)[k];

                    sum[k] = saturateCastUchar(top + bottom + center + left + right);
                }

                Result.put(j, i, sum);
            }
        }
      //! [basic_method_loop]

      //! [borders]
        Result.row(0).setTo(new Scalar(0));
        Result.row(Result.rows()-1).setTo(new Scalar(0));
        Result.col(0).setTo(new Scalar(0));
        Result.col(Result.cols()-1).setTo(new Scalar(0));
      //! [borders]

        return Result;
    }
    //! [basic_method]

    public Image toBufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte [] b = new byte[bufferSize];
        m.get(0,0,b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    public void displayImage(String title, Image img, int x, int y)
    {
        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame(title);
        JLabel lbl=new JLabel(icon);
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
        new MatMaskOperationsRun().run();
    }
}
