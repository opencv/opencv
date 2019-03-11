import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;
import java.util.*;

class DiscreteFourierTransformRun{
    private void help() {
        System.out.println("" +
                "This program demonstrated the use of the discrete Fourier transform (DFT). \n" +
                "The dft of an image is taken and it's power spectrum is displayed.\n" +
                "Usage:\n" +
                "./DiscreteFourierTransform [image_name -- default ../data/lena.jpg]");
    }

    public void run(String[] args){

        help();

        String filename = ((args.length > 0) ? args[0] : "../data/lena.jpg");

        Mat I = Imgcodecs.imread(filename, Imgcodecs.IMREAD_GRAYSCALE);
        if( I.empty() ) {
            System.out.println("Error opening image");
            System.exit(-1);
        }

        //! [expand]
        Mat padded = new Mat();                     //expand input image to optimal size
        int m = Core.getOptimalDFTSize( I.rows() );
        int n = Core.getOptimalDFTSize( I.cols() ); // on the border add zero values
        Core.copyMakeBorder(I, padded, 0, m - I.rows(), 0, n - I.cols(), Core.BORDER_CONSTANT, Scalar.all(0));
        //! [expand]

        //! [complex_and_real]
        List<Mat> planes = new ArrayList<Mat>();
        padded.convertTo(padded, CvType.CV_32F);
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexI = new Mat();
        Core.merge(planes, complexI);         // Add to the expanded another plane with zeros
        //! [complex_and_real]

        //! [dft]
        Core.dft(complexI, complexI);         // this way the result may fit in the source matrix
        //! [dft]

        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        //! [magnitude]
        Core.split(complexI, planes);                               // planes.get(0) = Re(DFT(I)
                                                                    // planes.get(1) = Im(DFT(I))
        Core.magnitude(planes.get(0), planes.get(1), planes.get(0));// planes.get(0) = magnitude
        Mat magI = planes.get(0);
        //! [magnitude]

        //! [log]
        Mat matOfOnes = Mat.ones(magI.size(), magI.type());
        Core.add(matOfOnes, magI, magI);         // switch to logarithmic scale
        Core.log(magI, magI);
        //! [log]

        //! [crop_rearrange]
        // crop the spectrum, if it has an odd number of rows or columns
        magI = magI.submat(new Rect(0, 0, magI.cols() & -2, magI.rows() & -2));

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = magI.cols()/2;
        int cy = magI.rows()/2;

        Mat q0 = new Mat(magI, new Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1 = new Mat(magI, new Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2 = new Mat(magI, new Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3 = new Mat(magI, new Rect(cx, cy, cx, cy)); // Bottom-Right

        Mat tmp = new Mat();               // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
        //! [crop_rearrange]

        magI.convertTo(magI, CvType.CV_8UC1);
        //! [normalize]
        Core.normalize(magI, magI, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1); // Transform the matrix with float values
                                                                            // into a viewable image form (float between
                                                                            // values 0 and 255).
        //! [normalize]

        HighGui.imshow("Input Image"       , I   );    // Show the result
        HighGui.imshow("Spectrum Magnitude", magI);
        HighGui.waitKey();

        System.exit(0);
    }
}


public class DiscreteFourierTransform {
    public static void main(String[] args) {
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new DiscreteFourierTransformRun().run(args);
    }
}
