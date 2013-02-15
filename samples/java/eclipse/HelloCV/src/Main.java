import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class Main {

    public static void main(String[] args) {
        System.loadLibrary("opencv_java244");
        Mat m  = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("m = " + m.dump());
    }

}
