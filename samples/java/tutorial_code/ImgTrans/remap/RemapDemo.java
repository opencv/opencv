import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Remap {
    private Mat mapX = new Mat();
    private Mat mapY = new Mat();
    private Mat dst = new Mat();
    private int ind = 0;

    //! [Update]
    private void updateMap() {
        float buffX[] = new float[(int) (mapX.total() * mapX.channels())];
        mapX.get(0, 0, buffX);

        float buffY[] = new float[(int) (mapY.total() * mapY.channels())];
        mapY.get(0, 0, buffY);

        for (int i = 0; i < mapX.rows(); i++) {
            for (int j = 0; j < mapX.cols(); j++) {
                switch (ind) {
                case 0:
                    if( j > mapX.cols()*0.25 && j < mapX.cols()*0.75 && i > mapX.rows()*0.25 && i < mapX.rows()*0.75 ) {
                        buffX[i*mapX.cols() + j] = 2*( j - mapX.cols()*0.25f ) + 0.5f;
                        buffY[i*mapY.cols() + j] = 2*( i - mapX.rows()*0.25f ) + 0.5f;
                    } else {
                        buffX[i*mapX.cols() + j] = 0;
                        buffY[i*mapY.cols() + j] = 0;
                    }
                    break;
                case 1:
                    buffX[i*mapX.cols() + j] = j;
                    buffY[i*mapY.cols() + j] = mapY.rows() - i;
                    break;
                case 2:
                    buffX[i*mapX.cols() + j] = mapY.cols() - j;
                    buffY[i*mapY.cols() + j] = i;
                    break;
                case 3:
                    buffX[i*mapX.cols() + j] = mapY.cols() - j;
                    buffY[i*mapY.cols() + j] = mapY.rows() - i;
                    break;
                default:
                    break;
                }
            }
        }
        mapX.put(0, 0, buffX);
        mapY.put(0, 0, buffY);
        ind = (ind+1) % 4;
    }
    //! [Update]

    public void run(String[] args) {
        String filename = args.length > 0 ? args[0] : "../data/chicky_512.png";
        //! [Load]
        Mat src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }
        //! [Load]

        //! [Create]
        mapX = new Mat(src.size(), CvType.CV_32F);
        mapY = new Mat(src.size(), CvType.CV_32F);
        //! [Create]

        //! [Window]
        final String winname = "Remap demo";
        HighGui.namedWindow(winname, HighGui.WINDOW_AUTOSIZE);
        //! [Window]

        //! [Loop]
        for (;;) {
            updateMap();
            Imgproc.remap(src, dst, mapX, mapY, Imgproc.INTER_LINEAR);
            HighGui.imshow(winname, dst);
            if (HighGui.waitKey(1000) == 27) {
                break;
            }
        }
        //! [Loop]
        System.exit(0);
    }
}

public class RemapDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        new Remap().run(args);
    }
}
