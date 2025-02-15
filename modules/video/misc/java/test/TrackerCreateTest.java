package org.opencv.test.video;

import java.io.File;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.DeepNeuralNet;
import org.opencv.test.OpenCVTestCase;

import org.opencv.video.Tracker;
import org.opencv.video.TrackerGOTURN;
import org.opencv.video.TrackerGOTURN_Params;
import org.opencv.video.TrackerNano;
import org.opencv.video.TrackerNano_Params;
import org.opencv.video.TrackerVit;
import org.opencv.video.TrackerVit_Params;
import org.opencv.video.TrackerMIL;

public class TrackerCreateTest extends OpenCVTestCase {

    private final static String ENV_OPENCV_DNN_TEST_DATA_PATH = "OPENCV_DNN_TEST_DATA_PATH";
    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";
    private String testDataPath;
    private String modelsDataPath;
    private Class netClass;
    private Class dnnClass;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        try {
            netClass = this.getClassForName("org.opencv.dnn.Net");
            dnnClass = this.getClassForName("org.opencv.dnn.Dnn");
        } catch (ClassNotFoundException e) {
            throw new TestSkipException();
        }

        // relys on https://developer.android.com/reference/java/lang/System
        isTestCaseEnabled = System.getProperties().getProperty("java.vm.name") != "Dalvik";
        if (!isTestCaseEnabled) {
            return;
        }

        testDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);
        if (testDataPath == null) {
            throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");
        }

        modelsDataPath = System.getenv(ENV_OPENCV_DNN_TEST_DATA_PATH);
        if (modelsDataPath == null) {
            modelsDataPath = testDataPath;
        }

        if (isTestCaseEnabled) {
            testDataPath = System.getenv(ENV_OPENCV_DNN_TEST_DATA_PATH);
            if (testDataPath == null)
                testDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);
            if (testDataPath == null)
                throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");
        }
    }

    public void testCreateTrackerGOTURN() {
        Object net;
        Tracker tracker;
        try {
            String protoFile = new File(testDataPath, "dnn/gsoc2016-goturn/goturn.prototxt").toString();
            String weightsFile = new File(modelsDataPath, "dnn/gsoc2016-goturn/goturn.caffemodel").toString();
            net = dnnClass.getMethod("readNetFromCaffe", String.class, String.class).invoke(protoFile, weightsFile);
            tracker = TrackerGOTURN.create(DeepNeuralNet.class.cast(net));
        } catch (CvException e) {
            return;
        } catch (Exception e) {
            return;
        }
        assert(tracker != null);
    }

    public void testCreateTrackerNano() {
        Object backbone;
        Object neckhead;
        Tracker tracker;
        try {
            String backboneFile = new File(modelsDataPath, "dnn/onnx/models/nanotrack_backbone_sim_v2.onnx").toString();
            String neckheadFile = new File(modelsDataPath, "dnn/onnx/models/nanotrack_head_sim_v2.onnx").toString();
            backbone = dnnClass.getMethod("readNet", String.class).invoke(backboneFile);
            neckhead = dnnClass.getMethod("readNet", String.class).invoke(neckheadFile);
            tracker = TrackerNano.create(DeepNeuralNet.class.cast(backbone), DeepNeuralNet.class.cast(neckhead));
        } catch (CvException e) {
            return;
        } catch (Exception e) {
            return;
        }
        assert(tracker != null);
    }

    public void testCreateTrackerVit() {
        Object net;
        Tracker tracker;
        try {
            String backboneFile = new File(modelsDataPath, "dnn/onnx/models/vitTracker.onnx").toString();
            net = dnnClass.getMethod("readNet", String.class).invoke(backboneFile);
            tracker = TrackerVit.create(DeepNeuralNet.class.cast(net));
        } catch (CvException e) {
            return;
        } catch (Exception e) {
            return;
        }
        assert(tracker != null);
    }

    public void testCreateTrackerMIL() {
        Tracker tracker = TrackerMIL.create();
        assert(tracker != null);
        Mat mat = new Mat(100, 100, CvType.CV_8UC1);
        Rect rect = new Rect(10, 10, 30, 30);
        tracker.init(mat, rect);  // should not crash (https://github.com/opencv/opencv/issues/19915)
    }
}
