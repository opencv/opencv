package org.opencv.test.dnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.DictValue;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Layer;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
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
    protected void setUp() throws Exception {
        super.setUp();

        String envDnnTestDataPath = System.getenv(ENV_OPENCV_DNN_TEST_DATA_PATH);

        if(envDnnTestDataPath == null){
            isTestCaseEnabled = false;
            return;
        }

        File dnnTestDataPath = new File(envDnnTestDataPath);
        modelFileName =  new File(dnnTestDataPath, "dnn/tensorflow_inception_graph.pb").toString();

        String envTestDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);

        if(envTestDataPath == null) throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");

        File testDataPath = new File(envTestDataPath);

        File f = new File(testDataPath, "dnn/grace_hopper_227.png");
        sourceImageFile = f.toString();
        if(!f.exists()) throw new Exception("Test image is missing: " + sourceImageFile);

        net = Dnn.readNetFromTensorflow(modelFileName);

        Mat image = Imgcodecs.imread(sourceImageFile);
        assertNotNull("Loading image from file failed!", image);

        Mat inputBlob = Dnn.blobFromImage(image, 1.0, new Size(224, 224), new Scalar(0), true, true);
        assertNotNull("Converting image to blob failed!", inputBlob);

        net.setInput(inputBlob, "input");
    }

    public void testSetInputsNames() {
        List<String> inputs = new ArrayList();
        inputs.add("input");
        try {
            net.setInputsNames(inputs);
        } catch(Exception e) {
            fail("Net setInputsNames failed: " + e.getMessage());
        }
    }

    public void testForward() {
        List<Mat> outs = new ArrayList();
        List<String> outNames = new ArrayList();
        outNames.add("softmax2");
        try {
            net.forward(outs,outNames);
        } catch(Exception e) {
            fail("Net forward failed: " + e.getMessage());
        }
    }

    public void testGetMemoryConsumption() {
        int layerId = 1;
        List<MatOfInt> netInputShapes = new ArrayList();
        netInputShapes.add(new MatOfInt(1, 3, 224, 224));
        long[] weights=null;
        long[] blobs=null;
        try {
            net.getMemoryConsumption(layerId, netInputShapes, weights, blobs);
        } catch(Exception e) {
            fail("Net getMemoryConsumption failed: " + e.getMessage());
        }
    }

    public void testGetFLOPS() {
        int layerId = 1;
        List<MatOfInt> netInputShapes = new ArrayList();
        netInputShapes.add(new MatOfInt(1, 3, 224, 224));
        try {
            net.getFLOPS(layerId, netInputShapes);
        } catch(Exception e) {
            fail("Net getFLOPS failed: " + e.getMessage());
        }
    }

    public void testGetLayersShapes() {
        List<MatOfInt> netInputShapes = new ArrayList();
        netInputShapes.add(new MatOfInt(1, 3, 224, 224));

        MatOfInt layersIds = new MatOfInt();
        List<List<MatOfInt>> inLayersShapes = new ArrayList();
        List<List<MatOfInt>> outLayersShapes = new ArrayList();
        try {
            net.getLayersShapes(netInputShapes, layersIds, inLayersShapes, outLayersShapes);

            assertEquals(layersIds.total(), inLayersShapes.size());
            assertEquals(layersIds.total(), outLayersShapes.size());

            // Layer ID for "conv2d0_pre_relu/conv"
            int layerId = 1;

            MatOfInt expectedInShape = new MatOfInt(1, 3, 224, 224);
            MatOfInt expectedOutShape = new MatOfInt(1, 64, 112, 112);

            // Test inLayersShapes
            MatOfInt actualInShape = inLayersShapes.get(layerId).get(0);
            assertMatEqual(expectedInShape, actualInShape);

            // Test outLayersShapes
            MatOfInt actualOutShape = outLayersShapes.get(layerId).get(0);
            assertMatEqual(expectedOutShape, actualOutShape);

        } catch(Exception e) {
            fail("Net getLayersShapes failed: " + e.getMessage());
        }
    }
}
