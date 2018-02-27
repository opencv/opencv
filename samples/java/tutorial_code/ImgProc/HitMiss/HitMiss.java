import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

class HitMissRun{

    public void run() {
        Mat input_image = new Mat( 8, 8, CvType.CV_8UC1 );
        int row = 0, col = 0;
        input_image.put(row ,col,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 255, 255, 255, 0, 0, 0, 255,
                0, 255, 255, 255, 0, 0, 0, 0,
                0, 255, 255, 255, 0, 255, 0, 0,
                0, 0, 255, 0, 0, 0, 0, 0,
                0, 0, 255, 0, 0, 255, 255, 0,
                0, 255, 0, 255, 0, 0, 255, 0,
                0, 255, 255, 255, 0, 0, 0, 0);

        Mat kernel = new Mat( 3, 3, CvType.CV_16S );
        kernel.put(row ,col,
                0, 1, 0,
                1, -1, 1,
                0, 1, 0 );

        Mat output_image = new Mat();
        Imgproc.morphologyEx(input_image, output_image, Imgproc.MORPH_HITMISS, kernel);

        int rate = 50;
        Core.add(kernel, new Scalar(1), kernel);
        Core.multiply(kernel, new Scalar(127), kernel);
        kernel.convertTo(kernel, CvType.CV_8U);

        Imgproc.resize(kernel, kernel, new Size(), rate, rate, Imgproc.INTER_NEAREST);
        HighGui.imshow("kernel", kernel);
        HighGui.moveWindow("kernel", 0, 0);

        Imgproc.resize(input_image, input_image, new Size(), rate, rate, Imgproc.INTER_NEAREST);
        HighGui.imshow("Original", input_image);
        HighGui.moveWindow("Original", 0, 200);

        Imgproc.resize(output_image, output_image, new Size(), rate, rate, Imgproc.INTER_NEAREST);
        HighGui.imshow("Hit or Miss", output_image);
        HighGui.moveWindow("Hit or Miss", 500, 200);

        HighGui.waitKey(0);
        System.exit(0);
    }
}

public class HitMiss
{
    public static void main(String[] args) {
        // load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new HitMissRun().run();
    }
}
