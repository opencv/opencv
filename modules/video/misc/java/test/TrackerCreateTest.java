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
import org.opencv.video.TrackerMIL;

public class TrackerCreateTest extends OpenCVTestCase {

    private final static String ENV_OPENCV_DNN_TEST_DATA_PATH = "OPENCV_DNN_TEST_DATA_PATH";
    private final static String ENV_OPENCV_TEST_DATA_PATH = "OPENCV_TEST_DATA_PATH";
    private String testDataPath;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        // relys on https://developer.android.com/reference/java/lang/System
        isTestCaseEnabled = System.getProperties().getProperty("java.vm.name") != "Dalvik";

        if (isTestCaseEnabled) {
            testDataPath = System.getenv(ENV_OPENCV_DNN_TEST_DATA_PATH);
            if (testDataPath == null)
                testDataPath = System.getenv(ENV_OPENCV_TEST_DATA_PATH);
            if (testDataPath == null)
                throw new Exception(ENV_OPENCV_TEST_DATA_PATH + " has to be defined!");
        }
    }

    public void testCreateTrackerGOTURNParams() {
        TrackerGOTURN_Params params = new TrackerGOTURN_Params();
        params.set_modelTxt(new File(testDataPath, "dnn/gsoc2016-goturn/goturn.prototxt").toString());
        params.set_modelBin(new File(testDataPath, "dnn/gsoc2016-goturn/goturn.caffemodel").toString());
        Tracker tracker = TrackerGOTURN.create(params);
        assert(tracker != null);
    }

    public void testCreateTrackerGOTURNModels() {
        Net net;
        try {
            String protoFile = new File(testDataPath, "dnn/gsoc2016-goturn/goturn.prototxt").toString();
            String weightsFile = new File(testDataPath, "dnn/gsoc2016-goturn/goturn.caffemodel").toString();
            net = Dnn.readNetFromCaffe(protoFile, weightsFile);
        } catch (CvException e) {
            return;
        }
        Tracker tracker = TrackerGOTURN.create(net);
        assert(tracker != null);
    }

    public void testCreateTrackerNanoParams() {
        TrackerNano_Params params = new TrackerNano_Params();
        params.set_backbone(new File(testDataPath, "dnn/onnx/models/nanotrack_backbone_sim_v2.onnx").toString());
        params.set_neckhead(new File(testDataPath, "dnn/onnx/models/nanotrack_head_sim_v2.onnx").toString());
        Tracker tracker = TrackerNano.create(params);
        assert(tracker != null);
    }

    public void testCreateTrackerNanoModels() {
        Net backbone;
        Net neckhead;
        try {
            String backboneFile = new File(testDataPath, "dnn/onnx/models/nanotrack_backbone_sim_v2.onnx").toString();
            String neckheadFile = new File(testDataPath, "dnn/onnx/models/nanotrack_head_sim_v2.onnx").toString();
            backbone = Dnn.readNet(backboneFile);
            neckhead = Dnn.readNet(neckheadFile);
        } catch (CvException e) {
            return;
        }
        Tracker tracker = TrackerNano.create(backbone, neckhead);
        assert(tracker != null);
    }

    public void testCreateTrackerVitParams() {
        TrackerVit_Params params = new TrackerVit_Params();
        params.set_net(new File(testDataPath, "dnn/onnx/models/vitTracker.onnx").toString());
        Tracker tracker = TrackerVit.create(params);
        assert(tracker != null);
    }

    public void testCreateTrackerVitModels() {
        Net net;
        try {
            String backboneFile = new File(testDataPath, "dnn/onnx/models/vitTracker.onnx").toString();
            net = Dnn.readNet(backboneFile);
        } catch (CvException e) {
            return;
        }
        Tracker tracker = TrackerVit.create(net);
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
