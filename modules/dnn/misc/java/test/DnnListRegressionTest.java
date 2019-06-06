package org.opencv.test.dnn;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeNotNull;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.test.OpenCVTestCase;

/*
*  regression test for #12324,
*    testing various java.util.List invocations,
*    which use the LIST_GET macro
*/

public class DnnListRegressionTest extends OpenCVTestCase {

    private final static String ENV_OPENCV_DNN_TEST_DATA_PATH = "OPENCV_DNN_TEST_DATA_PATH";

    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";

    String modelFileName = "";
    String sourceImageFile = "";

    Net net;

    @Override
    public void setUp() {
        super.setUp();

        String envDnnTestDataPath = System.getenv(ENV_OPENCV_DNN_TEST_DATA_PATH);
        assumeNotNull(envDnnTestDataPath);

        File dnnTestDataPath = new File(envDnnTestDataPath);
        modelFileName =  new File(dnnTestDataPath, "dnn/tensorflow_inception_graph.pb").toString();

        String envTestDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);

        assertNotNull(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!", envTestDataPath);

        File testDataPath = new File(envTestDataPath);

        File f = new File(testDataPath, "dnn/grace_hopper_227.png");
        sourceImageFile = f.toString();
        
        assertTrue("Test image is missing: " + sourceImageFile, f.exists());

        net = Dnn.readNetFromTensorflow(modelFileName);

        Mat image = Imgcodecs.imread(sourceImageFile);
        assertNotNull("Loading image from file failed!", image);

        Mat inputBlob = Dnn.blobFromImage(image, 1.0, new Size(224, 224), new Scalar(0), true, true);
        assertNotNull("Converting image to blob failed!", inputBlob);

        net.setInput(inputBlob, "input");
    }

    @Test
    public void testSetInputsNames() {
        List<String> inputs = new ArrayList<String>();
        inputs.add("input");
        try {
            net.setInputsNames(inputs);
        } catch(Exception e) {
            fail("Net setInputsNames failed: " + e.getMessage());
        }
    }

    @Test
    public void testForward() {
        List<Mat> outs = new ArrayList<Mat>();
        List<String> outNames = new ArrayList<String>();
        outNames.add("softmax2");
        try {
            net.forward(outs,outNames);
        } catch(Exception e) {
            fail("Net forward failed: " + e.getMessage());
        }
    }

    @Test
    public void testGetMemoryConsumption() {
        int layerId = 1;
        List<MatOfInt> netInputShapes = new ArrayList<MatOfInt>();
        netInputShapes.add(new MatOfInt(1, 3, 224, 224));
        long[] weights=null;
        long[] blobs=null;
        try {
            net.getMemoryConsumption(layerId, netInputShapes, weights, blobs);
        } catch(Exception e) {
            fail("Net getMemoryConsumption failed: " + e.getMessage());
        }
    }

    @Test
    public void testGetFLOPS() {
        int layerId = 1;
        List<MatOfInt> netInputShapes = new ArrayList<MatOfInt>();
        netInputShapes.add(new MatOfInt(1, 3, 224, 224));
        try {
            net.getFLOPS(layerId, netInputShapes);
        } catch(Exception e) {
            fail("Net getFLOPS failed: " + e.getMessage());
        }
    }
}
