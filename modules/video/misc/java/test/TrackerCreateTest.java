package org.opencv.test.video;

import java.io.File;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.test.OpenCVTestCase;

import org.opencv.video.Tracker;
import org.opencv.video.TrackerNano;
import org.opencv.video.TrackerNano_Params;
import org.opencv.video.TrackerVit;
import org.opencv.video.TrackerVit_Params;
import org.opencv.video.TrackerMIL;
import org.opencv.video.MultiTracker;
import org.opencv.video.MultiTracker_Params;

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

    public void testCreateTrackerMIL() {
        Tracker tracker = TrackerMIL.create();
        assert(tracker != null);
        Mat mat = new Mat(100, 100, CvType.CV_8UC1);
        Rect rect = new Rect(10, 10, 30, 30);
        tracker.init(mat, rect);  // should not crash (https://github.com/opencv/opencv/issues/19915)
    }

    public void testCreateMultiTracker() {
        MultiTracker_Params params = new MultiTracker_Params();
        params.set_minHits(1);   // reported on the frame it is created
        MultiTracker tracker = MultiTracker.create(params);
        assert(tracker != null);

        MatOfRect2d detBoxes = new MatOfRect2d(new Rect2d(10, 10, 20, 40));
        MatOfFloat detScores = new MatOfFloat(0.9f);
        MatOfInt detClassIds = new MatOfInt(0);

        MatOfInt trackIds = new MatOfInt();
        MatOfRect2d trackBoxes = new MatOfRect2d();
        MatOfInt trackClassIds = new MatOfInt();

        tracker.update(detBoxes, detScores, detClassIds, trackIds, trackBoxes, trackClassIds);

        assertEquals(1, trackIds.toList().size());
        assertEquals(1, trackBoxes.toList().size());
        assertEquals(0, trackClassIds.toList().get(0).intValue());

        tracker.reset();
    }
}
