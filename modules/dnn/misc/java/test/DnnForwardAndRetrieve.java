package org.opencv.test.dnn;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.test.OpenCVTestCase;

public class DnnForwardAndRetrieve extends OpenCVTestCase {

    private final static String ENV_OPENCV_DNN_TEST_DATA_PATH = "OPENCV_DNN_TEST_DATA_PATH";
    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";

    private String modelFileName = "";

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        String dnnTestDataPath = System.getenv(ENV_OPENCV_DNN_TEST_DATA_PATH);
        String generalTestDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);

        File model = null;

        if (generalTestDataPath != null) {
            model = new File(generalTestDataPath, "dnn/onnx/models/split_0.onnx");
        }

        if ((model == null || !model.isFile()) && dnnTestDataPath != null) {
            model = new File(dnnTestDataPath, "dnn/onnx/models/split_0.onnx");
        }

        if (model == null || !model.isFile()) {
            isTestCaseEnabled = false;
            return;
        }

        modelFileName = model.getAbsolutePath();
    }

    public void testForwardAndRetrieve()
    {
        // Verifies forwardAndRetrieve nested list marshalling using a small ONNX model instead of the removed Caffe importer.
        Net net = Dnn.readNetFromONNX(modelFileName, Dnn.ENGINE_CLASSIC);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);

        // split_0.onnx declares a single 4D input named "image" of shape [1, 3, 2, 2].
        Mat inp = new Mat(new int[]{1, 3, 2, 2}, CvType.CV_32F);
        Core.randu(inp, -1, 1);
        net.setInput(inp);

        List<String> outNames = net.getUnconnectedOutLayersNames();
        assertFalse("Model has no output layers", outNames.isEmpty());

        // Forward and retrieve every output blob of the requested layers.
        List<List<Mat>> outBlobs = new ArrayList<>();
        net.forwardAndRetrieve(outBlobs, outNames);

        // One entry per requested layer name, each holding at least one valid blob.
        assertEquals(outNames.size(), outBlobs.size());
        for (List<Mat> blobs : outBlobs) {
            assertFalse(blobs.isEmpty());
            for (Mat blob : blobs)
                assertFalse(blob.empty());
        }
    }
}
