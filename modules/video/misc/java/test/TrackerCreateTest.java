package org.opencv.test.video;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.test.OpenCVTestCase;

import org.opencv.video.Tracker;
import org.opencv.video.TrackerGOTURN;
import org.opencv.video.TrackerMIL;

public class TrackerCreateTest extends OpenCVTestCase {

    @Override
    protected void setUp() throws Exception {
        super.setUp();
    }


    public void testCreateTrackerGOTURN() {
        try {
            Tracker tracker = TrackerGOTURN.create();
            assert(tracker != null);
        } catch (CvException e) {
            // expected, model files may be missing
        }
    }

    public void testCreateTrackerMIL() {
        Tracker tracker = TrackerMIL.create();
        assert(tracker != null);
        Mat mat = new Mat(100, 100, CvType.CV_8UC1);
        Rect rect = new Rect(10, 10, 30, 30);
        tracker.init(mat, rect);  // should not crash (https://github.com/opencv/opencv/issues/19915)
    }

}
