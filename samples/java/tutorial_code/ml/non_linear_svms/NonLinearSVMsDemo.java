import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;

public class NonLinearSVMsDemo {
    public static final int NTRAINING_SAMPLES = 100;
    public static final float FRAC_LINEAR_SEP = 0.9f;

    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        System.out.println("\n--------------------------------------------------------------------------");
        System.out.println("This program shows Support Vector Machines for Non-Linearly Separable Data. ");
        System.out.println("--------------------------------------------------------------------------\n");

        // Data for visual representation
        int width = 512, height = 512;
        Mat I = Mat.zeros(height, width, CvType.CV_8UC3);

        // --------------------- 1. Set up training data randomly---------------------------------------
        Mat trainData = new Mat(2 * NTRAINING_SAMPLES, 2, CvType.CV_32F);
        Mat labels = new Mat(2 * NTRAINING_SAMPLES, 1, CvType.CV_32S);

        Random rng = new Random(100); // Random value generation class

        // Set up the linearly separable part of the training data
        int nLinearSamples = (int) (FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

        //! [setup1]
        // Generate random points for the class 1
        Mat trainClass = trainData.rowRange(0, nLinearSamples);
        // The x coordinate of the points is in [0, 0.4)
        Mat c = trainClass.colRange(0, 1);
        float[] cData = new float[(int) (c.total() * c.channels())];
        double[] cDataDbl = rng.doubles(cData.length, 0, 0.4f * width).toArray();
        for (int i = 0; i < cData.length; i++) {
            cData[i] = (float) cDataDbl[i];
        }
        c.put(0, 0, cData);
        // The y coordinate of the points is in [0, 1)
        c = trainClass.colRange(1, 2);
        cData = new float[(int) (c.total() * c.channels())];
        cDataDbl = rng.doubles(cData.length, 0, height).toArray();
        for (int i = 0; i < cData.length; i++) {
            cData[i] = (float) cDataDbl[i];
        }
        c.put(0, 0, cData);

        // Generate random points for the class 2
        trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
        // The x coordinate of the points is in [0.6, 1]
        c = trainClass.colRange(0, 1);
        cData = new float[(int) (c.total() * c.channels())];
        cDataDbl = rng.doubles(cData.length, 0.6 * width, width).toArray();
        for (int i = 0; i < cData.length; i++) {
            cData[i] = (float) cDataDbl[i];
        }
        c.put(0, 0, cData);
        // The y coordinate of the points is in [0, 1)
        c = trainClass.colRange(1, 2);
        cData = new float[(int) (c.total() * c.channels())];
        cDataDbl = rng.doubles(cData.length, 0, height).toArray();
        for (int i = 0; i < cData.length; i++) {
            cData[i] = (float) cDataDbl[i];
        }
        c.put(0, 0, cData);
        //! [setup1]

        // ------------------ Set up the non-linearly separable part of the training data ---------------
        //! [setup2]
        // Generate random points for the classes 1 and 2
        trainClass = trainData.rowRange(nLinearSamples, 2 * NTRAINING_SAMPLES - nLinearSamples);
        // The x coordinate of the points is in [0.4, 0.6)
        c = trainClass.colRange(0, 1);
        cData = new float[(int) (c.total() * c.channels())];
        cDataDbl = rng.doubles(cData.length, 0.4 * width, 0.6 * width).toArray();
        for (int i = 0; i < cData.length; i++) {
            cData[i] = (float) cDataDbl[i];
        }
        c.put(0, 0, cData);
        // The y coordinate of the points is in [0, 1)
        c = trainClass.colRange(1, 2);
        cData = new float[(int) (c.total() * c.channels())];
        cDataDbl = rng.doubles(cData.length, 0, height).toArray();
        for (int i = 0; i < cData.length; i++) {
            cData[i] = (float) cDataDbl[i];
        }
        c.put(0, 0, cData);
        //! [setup2]

        // ------------------------- Set up the labels for the classes---------------------------------
        labels.rowRange(0, NTRAINING_SAMPLES).setTo(new Scalar(1)); // Class 1
        labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(new Scalar(2)); // Class 2

        // ------------------------ 2. Set up the support vector machines parameters--------------------
        System.out.println("Starting training process");
        //! [init]
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setC(0.1);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, (int) 1e7, 1e-6));
        //! [init]

        // ------------------------ 3. Train the svm----------------------------------------------------
        //! [train]
        svm.train(trainData, Ml.ROW_SAMPLE, labels);
        //! [train]
        System.out.println("Finished training process");

        // ------------------------ 4. Show the decision regions----------------------------------------
        //! [show]
        byte[] IData = new byte[(int) (I.total() * I.channels())];
        Mat sampleMat = new Mat(1, 2, CvType.CV_32F);
        float[] sampleMatData = new float[(int) (sampleMat.total() * sampleMat.channels())];
        for (int i = 0; i < I.rows(); i++) {
            for (int j = 0; j < I.cols(); j++) {
                sampleMatData[0] = j;
                sampleMatData[1] = i;
                sampleMat.put(0, 0, sampleMatData);
                float response = svm.predict(sampleMat);

                if (response == 1) {
                    IData[(i * I.cols() + j) * I.channels()] = 0;
                    IData[(i * I.cols() + j) * I.channels() + 1] = 100;
                    IData[(i * I.cols() + j) * I.channels() + 2] = 0;
                } else if (response == 2) {
                    IData[(i * I.cols() + j) * I.channels()] = 100;
                    IData[(i * I.cols() + j) * I.channels() + 1] = 0;
                    IData[(i * I.cols() + j) * I.channels() + 2] = 0;
                }
            }
        }
        I.put(0, 0, IData);
        //! [show]

        // ----------------------- 5. Show the training data--------------------------------------------
        //! [show_data]
        int thick = -1;
        int lineType = Core.LINE_8;
        float px, py;
        // Class 1
        float[] trainDataData = new float[(int) (trainData.total() * trainData.channels())];
        trainData.get(0, 0, trainDataData);
        for (int i = 0; i < NTRAINING_SAMPLES; i++) {
            px = trainDataData[i * trainData.cols()];
            py = trainDataData[i * trainData.cols() + 1];
            Imgproc.circle(I, new Point(px, py), 3, new Scalar(0, 255, 0), thick, lineType, 0);
        }
        // Class 2
        for (int i = NTRAINING_SAMPLES; i < 2 * NTRAINING_SAMPLES; ++i) {
            px = trainDataData[i * trainData.cols()];
            py = trainDataData[i * trainData.cols() + 1];
            Imgproc.circle(I, new Point(px, py), 3, new Scalar(255, 0, 0), thick, lineType, 0);
        }
        //! [show_data]

        // ------------------------- 6. Show support vectors--------------------------------------------
        //! [show_vectors]
        thick = 2;
        Mat sv = svm.getUncompressedSupportVectors();
        float[] svData = new float[(int) (sv.total() * sv.channels())];
        sv.get(0, 0, svData);
        for (int i = 0; i < sv.rows(); i++) {
            Imgproc.circle(I, new Point(svData[i * sv.cols()], svData[i * sv.cols() + 1]), 6, new Scalar(128, 128, 128),
                    thick, lineType, 0);
        }
        //! [show_vectors]

        Imgcodecs.imwrite("result.png", I); // save the Image
        HighGui.imshow("SVM for Non-Linear Training Data", I); // show it to the user
        HighGui.waitKey();
        System.exit(0);
    }
}
