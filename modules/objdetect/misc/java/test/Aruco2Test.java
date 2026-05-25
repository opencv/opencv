package org.opencv.test.aruco2;

import java.util.List;
import org.opencv.test.OpenCVTestCase;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Aruco2;
import org.opencv.objdetect.DetectionParameters;
import org.opencv.objdetect.Diamond;
import org.opencv.objdetect.FiducialMarker;
import org.opencv.objdetect.FractalMarker;
import org.opencv.objdetect.Aruco2_GridBoard;

public class Aruco2Test extends OpenCVTestCase {

    public void testDetection() {
        DetectionParameters params = new DetectionParameters();
        params.set_boxFilterSize(5);

        Mat img = new Mat();
        Aruco2.getFiducialMarkerImage(img, Aruco2.DICT_4X4_50, 0, 20, true);
        assertFalse(img.empty());

        List<FiducialMarker> markers = Aruco2.detectFiducialMarkers(img, Aruco2.DICT_4X4_50, params);
        assertEquals(1, markers.size());
        assertEquals(0, markers.get(0).get_id());
        assertEquals(Aruco2.DICT_4X4_50, markers.get(0).get_dictionary());
    }

    public void testGenerateMarker() {
        Mat img = new Mat();
        Aruco2.getFiducialMarkerImage(img, Aruco2.DICT_4X4_50, 1, 20, false);
        assertFalse(img.empty());
        assertEquals(120, img.rows());
        assertEquals(120, img.cols());
    }

    public void testGridBoard() {
        Mat img = new Mat();
        Size gridSize = new Size(3, 3);
        Aruco2.getGridBoardImage(img, gridSize, Aruco2.DICT_4X4_50);
        assertFalse(img.empty());
    }

    public void testDrawFiducialMarkers() {
        Mat img = new Mat();
        Aruco2.getFiducialMarkerImage(img, Aruco2.DICT_ARUCO_MIP_36h12, 42, 20, true);
        assertFalse(img.empty());

        List<FiducialMarker> markers = Aruco2.detectFiducialMarkers(img, Aruco2.DICT_ARUCO_MIP_36h12);
        assertEquals(1, markers.size());

        Mat colorImg = new Mat();
        Imgproc.cvtColor(img, colorImg, Imgproc.COLOR_GRAY2BGR);
        Aruco2.drawFiducialMarkers(colorImg, markers);
        assertFalse(colorImg.empty());
    }

    public void testDetectFiducialMarkersMultiDict() {
        Mat img = new Mat();
        Aruco2.getFiducialMarkerImage(img, Aruco2.DICT_4X4_50, 0, 20, true);
        assertFalse(img.empty());

        MatOfInt dicts = new MatOfInt(Aruco2.DICT_4X4_50, Aruco2.DICT_4X4_100);
        List<FiducialMarker> markers = Aruco2.detectFiducialMarkers(img, dicts, new DetectionParameters());
        assertEquals(1, markers.size());
        assertEquals(0, markers.get(0).get_id());
    }

    public void testDrawAxis() {
        Mat img = new Mat();
        Aruco2.getFiducialMarkerImage(img, Aruco2.DICT_ARUCO_MIP_36h12, 42, 20, true);
        Mat colorImg = new Mat();
        Imgproc.cvtColor(img, colorImg, Imgproc.COLOR_GRAY2BGR);

        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, 500, 0, 200, 0, 500, 200, 0, 0, 1);
        Mat distCoeffs = Mat.zeros(4, 1, CvType.CV_64F);
        Mat rvec = Mat.zeros(3, 1, CvType.CV_64F);
        Mat tvec = new Mat(3, 1, CvType.CV_64F);
        tvec.put(0, 0, 0, 0, 1);

        Aruco2.drawAxis(colorImg, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);
        assertFalse(colorImg.empty());
    }

    public void testGetSolvePnpPointsFiducialMarker() {
        Mat img = new Mat();
        Aruco2.getFiducialMarkerImage(img, Aruco2.DICT_ARUCO_MIP_36h12, 100, 20, false);
        Mat canvas = new Mat(img.rows() * 2, img.cols() * 2, CvType.CV_8UC1, new Scalar(255));
        img.copyTo(canvas.submat(img.rows() / 2, img.rows() / 2 + img.rows(),
                                 img.cols() / 2, img.cols() / 2 + img.cols()));

        List<FiducialMarker> markers = Aruco2.detectFiducialMarkers(canvas, Aruco2.DICT_ARUCO_MIP_36h12);
        assertEquals(1, markers.size());

        Mat objPoints = new Mat();
        Mat imgPoints = new Mat();
        Aruco2.getSolvePnpPoints(markers.get(0), objPoints, imgPoints);
        assertEquals(4, objPoints.total());
        assertEquals(4, imgPoints.total());
    }

    public void testDetectAndDrawGridBoard() {
        Mat boardImg = new Mat();
        Size gridSize = new Size(3, 2);
        Aruco2.getGridBoardImage(boardImg, gridSize, Aruco2.DICT_ARUCO_MIP_36h12, 20);
        Mat canvas = new Mat(boardImg.rows() + 100, boardImg.cols() + 100, CvType.CV_8UC1, new Scalar(255));
        boardImg.copyTo(canvas.submat(50, 50 + boardImg.rows(), 50, 50 + boardImg.cols()));

        Aruco2_GridBoard board = new Aruco2_GridBoard();
        boolean found = Aruco2.detectGridBoard(canvas, gridSize, Aruco2.DICT_ARUCO_MIP_36h12, board);
        assertTrue(found);
        assertEquals(Aruco2.DICT_ARUCO_MIP_36h12, board.get_dictionary());
        assertEquals(6, board.get_markers().size()); // 3x2 = 6 markers

        Mat colorCanvas = new Mat();
        Imgproc.cvtColor(canvas, colorCanvas, Imgproc.COLOR_GRAY2BGR);
        Aruco2.drawGridBoard(colorCanvas, board);
        assertFalse(colorCanvas.empty());
    }

    public void testGetSolvePnpPointsGridBoard() {
        Mat boardImg = new Mat();
        Size gridSize = new Size(3, 2);
        Aruco2.getGridBoardImage(boardImg, gridSize, Aruco2.DICT_ARUCO_MIP_36h12, 20);
        Mat canvas = new Mat(boardImg.rows() + 100, boardImg.cols() + 100, CvType.CV_8UC1, new Scalar(255));
        boardImg.copyTo(canvas.submat(50, 50 + boardImg.rows(), 50, 50 + boardImg.cols()));

        Aruco2_GridBoard board = new Aruco2_GridBoard();
        boolean found = Aruco2.detectGridBoard(canvas, gridSize, Aruco2.DICT_ARUCO_MIP_36h12, board);
        assertTrue(found);

        Mat objPoints = new Mat();
        Mat imgPoints = new Mat();
        Aruco2.getSolvePnpPoints(board, objPoints, imgPoints, 0.05f);
        // 3x2 board → (3+1)×(2+1) = 12 intersection corners
        assertEquals(12, objPoints.total());
        assertEquals(12, imgPoints.total());
    }

    public void testDiamondWorkflow() {
        Mat diamondImg = new Mat();
        int[] ids = {5, 10, 15, 20};
        Aruco2.getDiamondImage(diamondImg, Aruco2.DICT_ARUCO_MIP_36h12, ids, 20);
        assertFalse(diamondImg.empty());

        Mat canvas = new Mat(diamondImg.rows() + 100, diamondImg.cols() + 100, CvType.CV_8UC1, new Scalar(255));
        diamondImg.copyTo(canvas.submat(50, 50 + diamondImg.rows(), 50, 50 + diamondImg.cols()));

        List<Diamond> diamonds = Aruco2.detectDiamonds(canvas, Aruco2.DICT_ARUCO_MIP_36h12);
        assertEquals(1, diamonds.size());
        assertEquals(Aruco2.DICT_ARUCO_MIP_36h12, diamonds.get(0).get_dictionary());

        Mat colorCanvas = new Mat();
        Imgproc.cvtColor(canvas, colorCanvas, Imgproc.COLOR_GRAY2BGR);
        Aruco2.drawDiamonds(colorCanvas, diamonds, new Scalar(0, 255, 0), false);
        assertFalse(colorCanvas.empty());

        Mat objPoints = new Mat();
        Mat imgPoints = new Mat();
        Aruco2.getSolvePnpPoints(diamonds.get(0), objPoints, imgPoints, 0.1f);
        // Diamond returns a 3x3 grid of 9 points
        assertEquals(9, objPoints.total());
        assertEquals(9, imgPoints.total());
    }

    public void testFractalWorkflow() {
        Mat fractalImg = new Mat();
        Aruco2.getFractalMarkerImage(fractalImg, Aruco2.FRACTAL_2L_6, 40);
        assertFalse(fractalImg.empty());
        assertEquals(fractalImg.rows(), fractalImg.cols());

        Mat canvas = new Mat(fractalImg.rows() + 100, fractalImg.cols() + 100, CvType.CV_8UC1, new Scalar(255));
        fractalImg.copyTo(canvas.submat(50, 50 + fractalImg.rows(), 50, 50 + fractalImg.cols()));

        List<FractalMarker> fractals = Aruco2.detectFractals(canvas, Aruco2.FRACTAL_2L_6);
        assertEquals(1, fractals.size());

        Mat colorCanvas = new Mat();
        Imgproc.cvtColor(canvas, colorCanvas, Imgproc.COLOR_GRAY2BGR);
        Aruco2.drawFractals(colorCanvas, fractals, new Scalar(0, 255, 0), true);
        assertFalse(colorCanvas.empty());

        Mat objPoints = new Mat();
        Mat imgPoints = new Mat();
        Aruco2.getSolvePnpPoints(fractals.get(0), objPoints, imgPoints, 0.2f);
        assertTrue(objPoints.total() >= 4);
        assertEquals(objPoints.total(), imgPoints.total());
    }
}
