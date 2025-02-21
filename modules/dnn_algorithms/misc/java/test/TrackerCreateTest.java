package org.opencv.test.video;

import java.io.File;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.test.OpenCVTestCase;

import org.opencv.video.Tracker;
import org.opencv.video.TrackerGOTURN;
import org.opencv.video.TrackerGOTURN_Params;
import org.opencv.video.TrackerNano;
import org.opencv.video.TrackerNano_Params;
import org.opencv.video.TrackerVit;
import org.opencv.video.TrackerVit_Params;

public class TrackerCreateTest extends OpenCVTestCase {

    private final static String ENV_OPENCV_DNN_TEST_DATA_PATH = "OPENCV_DNN_TEST_DATA_PATH";
    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";
    private String testDataPath;
    private String modelsDataPath;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

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
        Net net;
        try {
            String protoFile = new File(testDataPath, "dnn/gsoc2016-goturn/goturn.prototxt").toString();
            String weightsFile = new File(modelsDataPath, "dnn/gsoc2016-goturn/goturn.caffemodel").toString();
            net = Dnn.readNetFromCaffe(protoFile, weightsFile);
        } catch (CvException e) {
            return;
        }
        Tracker tracker = TrackerGOTURN.create(net);
        assert(tracker != null);
    }

    public void testCreateTrackerNano() {
        Net backbone;
        Net neckhead;
        try {
            String backboneFile = new File(modelsDataPath, "dnn/onnx/models/nanotrack_backbone_sim_v2.onnx").toString();
            String neckheadFile = new File(modelsDataPath, "dnn/onnx/models/nanotrack_head_sim_v2.onnx").toString();
            backbone = Dnn.readNet(backboneFile);
            neckhead = Dnn.readNet(neckheadFile);
        } catch (CvException e) {
            return;
        }
        Tracker tracker = TrackerNano.create(backbone, neckhead);
        assert(tracker != null);
    }

    public void testCreateTrackerVit() {
        Net net;
        try {
            String backboneFile = new File(modelsDataPath, "dnn/onnx/models/vitTracker.onnx").toString();
            net = Dnn.readNet(backboneFile);
        } catch (CvException e) {
            return;
        }
        Tracker tracker = TrackerVit.create(net);
        assert(tracker != null);
    }

}
