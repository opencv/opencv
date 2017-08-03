package org.opencv.test.dnn;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.DictValue;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Importer;
import org.opencv.dnn.Layer;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.test.OpenCVTestCase;

public class DnnTensorFlowTest extends OpenCVTestCase {

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

        File f = new File(testDataPath, "dnn/space_shuttle.jpg");
        sourceImageFile = f.toString();
        if(!f.exists()) throw new Exception("Test image is missing: " + sourceImageFile);

        net = Dnn.readNetFromTensorflow(modelFileName);
    }

    public void testGetLayerTypes() {
        List<String> layertypes = new ArrayList();
        net.getLayerTypes(layertypes);

        assertFalse("No layer types returned!", layertypes.isEmpty());
    }

    public void testGetLayer() {
        List<String> layernames = net.getLayerNames();

        assertFalse("Test net returned no layers!", layernames.isEmpty());

        String testLayerName = layernames.get(0);

        DictValue layerId = new DictValue(testLayerName);

        assertEquals("DictValue did not return the string, which was used in constructor!", testLayerName, layerId.getStringValue());

        Layer layer = net.getLayer(layerId);

        assertEquals("Layer name does not match the expected value!", testLayerName, layer.get_name());

    }

    public void testTestNetForward() {
        Mat rawImage = Imgcodecs.imread(sourceImageFile);

        assertNotNull("Loading image from file failed!", rawImage);

        Mat image = new Mat();
        Imgproc.resize(rawImage, image, new Size(224,224));

        Mat inputBlob = Dnn.blobFromImage(image);
        assertNotNull("Converting image to blob failed!", inputBlob);

        Mat inputBlobP = new Mat();
        Core.subtract(inputBlob, new Scalar(117.0), inputBlobP);

        net.setInput(inputBlobP, "input" );

        Mat result = net.forward();

        assertNotNull("Net returned no result!", result);

        Core.MinMaxLocResult minmax = Core.minMaxLoc(result.reshape(1, 1));

        assertTrue("No image recognized!", minmax.maxVal > 0.9);


    }

}
