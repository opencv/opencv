import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

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

        HighGui.namedWindow("Input", HighGui.WINDOW_AUTOSIZE);
        HighGui.namedWindow("Output", HighGui.WINDOW_AUTOSIZE);

        HighGui.imshow( "Input", src );
        double t = System.currentTimeMillis();

        Mat dst0 = sharpen(src, new Mat());

        t = ((double) System.currentTimeMillis() - t) / 1000;
        System.out.println("Hand written function time passed in seconds: " + t);

        HighGui.imshow( "Output", dst0 );
        HighGui.moveWindow("Output", 400, 400);
        HighGui.waitKey();

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

        HighGui.imshow( "Output", dst1 );

        HighGui.waitKey();
        System.exit(0);
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
}

public class MatMaskOperations {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new MatMaskOperationsRun().run(args);
    }
}
