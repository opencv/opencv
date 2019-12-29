import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class DisplayImage{
    public void run(String[] args){

        String filename = args.length > 0 ? args[0] : "../data/lena.jpg";

        // Create the object for the image
        Mat image=new Mat();

        //Read the image from the disk
        image=Imgcodecs.imread(filename,Imgcodecs.IMREAD_COLOR);

        //Exit if the image is not found
        if(image.empty()){
            System.out.println("Cannot load the image "+filename);
            System.exit(-1);
        }

        //Show the image window
        HighGui.imshow("Display Window",image);

        //Wait for a keystroke in the window
        HighGui.waitKey(0);
        System.exit(0);
    }
}

public class DisplayImageDemo {
    public static void main(String args[]){
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new DisplayImage().run(args);
    }
}
