import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class MatOperations {
    @SuppressWarnings("unused")
    public static void main(String[] args) {
        /*  Snippet code for Operations with images tutorial (not intended to be run) */

        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String filename = "";
        // Input/Output
        {
            //! [Load an image from a file]
            Mat img = Imgcodecs.imread(filename);
            //! [Load an image from a file]
        }
        {
            //! [Load an image from a file in grayscale]
            Mat img = Imgcodecs.imread(filename, Imgcodecs.IMREAD_GRAYSCALE);
            //! [Load an image from a file in grayscale]
        }
        {
            Mat img = new Mat(4, 4, CvType.CV_8U);
            //! [Save image]
            Imgcodecs.imwrite(filename, img);
            //! [Save image]
        }
        // Accessing pixel intensity values
        {
            Mat img = new Mat(4, 4, CvType.CV_8U);
            int y = 0, x = 0;
            {
                //! [Pixel access 1]
                byte[] imgData = new byte[(int) (img.total() * img.channels())];
                img.get(0, 0, imgData);
                byte intensity = imgData[y * img.cols() + x];
                //! [Pixel access 1]
            }
            {
                //! [Pixel access 5]
                byte[] imgData = new byte[(int) (img.total() * img.channels())];
                imgData[y * img.cols() + x] = (byte) 128;
                img.put(0, 0, imgData);
                //! [Pixel access 5]
            }

        }
        // Memory management and reference counting
        {
            //! [Reference counting 2]
            Mat img = Imgcodecs.imread("image.jpg");
            Mat img1 = img.clone();
            //! [Reference counting 2]
        }
        {
            //! [Reference counting 3]
            Mat img = Imgcodecs.imread("image.jpg");
            Mat sobelx = new Mat();
            Imgproc.Sobel(img, sobelx, CvType.CV_32F, 1, 0);
            //! [Reference counting 3]
        }
        // Primitive operations
        {
            Mat img = new Mat(400, 400, CvType.CV_8UC3);
            {
                //! [Set image to black]
                byte[] imgData = new byte[(int) (img.total() * img.channels())];
                Arrays.fill(imgData, (byte) 0);
                img.put(0, 0, imgData);
                //! [Set image to black]
            }
            {
                //! [Select ROI]
                Rect r = new Rect(10, 10, 100, 100);
                Mat smallImg = img.submat(r);
                //! [Select ROI]
            }
        }
        {
            //! [BGR to Gray]
            Mat img = Imgcodecs.imread("image.jpg"); // loading a 8UC3 image
            Mat grey = new Mat();
            Imgproc.cvtColor(img, grey, Imgproc.COLOR_BGR2GRAY);
            //! [BGR to Gray]
        }
        {
            Mat dst = new Mat(), src = new Mat();
            //! [Convert to CV_32F]
            src.convertTo(dst, CvType.CV_32F);
            //! [Convert to CV_32F]
        }
        // Visualizing images
        {
            //! [imshow 1]
            Mat img = Imgcodecs.imread("image.jpg");
            HighGui.namedWindow("image", HighGui.WINDOW_AUTOSIZE);
            HighGui.imshow("image", img);
            HighGui.waitKey();
            //! [imshow 1]
        }
        {
            //! [imshow 2]
            Mat img = Imgcodecs.imread("image.jpg");
            Mat grey = new Mat();
            Imgproc.cvtColor(img, grey, Imgproc.COLOR_BGR2GRAY);
            Mat sobelx = new Mat();
            Imgproc.Sobel(grey, sobelx, CvType.CV_32F, 1, 0);
            MinMaxLocResult res = Core.minMaxLoc(sobelx); // find minimum and maximum intensities
            Mat draw = new Mat();
            double maxVal = res.maxVal, minVal = res.minVal;
            sobelx.convertTo(draw, CvType.CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            HighGui.namedWindow("image", HighGui.WINDOW_AUTOSIZE);
            HighGui.imshow("image", draw);
            HighGui.waitKey();
            //! [imshow 2]
        }
        System.exit(0);
    }

}
