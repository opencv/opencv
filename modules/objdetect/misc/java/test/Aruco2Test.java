package org.opencv.test.aruco2;

import java.util.List;
import org.opencv.test.OpenCVTestCase;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.objdetect.Aruco2;
import org.opencv.objdetect.DetectionParameters;
import org.opencv.objdetect.FiducialMarker;

public class Aruco2Test extends OpenCVTestCase {

    public void testDetection() {
        DetectionParameters params = new DetectionParameters();
        params.set_boxFilterSize(5);

        Mat img = new Mat();
        Aruco2.getFiducialMarker(img, Aruco2.DICT_4X4_50, 0, 20, true);
        assertFalse(img.empty());

        List<FiducialMarker> markers = Aruco2.detectFiducialMarkers(img, Aruco2.DICT_4X4_50, params);
        assertEquals(1, markers.size());
        assertEquals(0, markers.get(0).get_id());
    }

    public void testGenerateMarker() {
        Mat img = new Mat();
        Aruco2.getFiducialMarker(img, Aruco2.DICT_4X4_50, 1, 20, false);
        assertFalse(img.empty());
        assertEquals(120, img.rows());
        assertEquals(120, img.cols());
    }

    public void testGridBoard() {
        Mat img = new Mat();
        Size gridSize = new Size(3, 3);
        Aruco2.getGridBoard(img, gridSize, Aruco2.DICT_4X4_50);
        assertFalse(img.empty());
    }
}
