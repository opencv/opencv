package org.opencv.test.aruco;

import java.util.ArrayList;
import java.util.List;

import org.opencv.test.OpenCVTestCase;
import org.junit.Assert;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.objdetect.*;


public class ArucoTest extends OpenCVTestCase {

    public void testGenerateBoards() {
        Dictionary dictionary = Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50);

        Mat point1 = new Mat(4, 3, CvType.CV_32FC1);
        int row = 0, col = 0;
        double squareLength = 40.;
        point1.put(row, col, 0, 0, 0,
                   0, squareLength, 0,
                   squareLength, squareLength, 0,
                   0, squareLength, 0);
        List<Mat>objPoints = new ArrayList<Mat>();
        objPoints.add(point1);

        Mat ids = new Mat(1, 1, CvType.CV_32SC1);
        ids.put(row, col, 0);

        Board board = new Board(objPoints, dictionary, ids);

        Mat image = new Mat();
        board.generateImage(new Size(80, 80), image, 2);

        assertTrue(image.total() > 0);
    }

    public void testArucoIssue3133() {
        byte[][] marker = {{0,1,1},{1,1,1},{0,1,1}};
        Dictionary dictionary = Objdetect.extendDictionary(1, 3);
        dictionary.set_maxCorrectionBits(0);
        Mat markerBits = new Mat(3, 3, CvType.CV_8UC1);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                markerBits.put(i, j, marker[i][j]);
            }
        }

        Mat markerCompressed = Dictionary.getByteListFromBits(markerBits);
        assertMatNotEqual(markerCompressed, dictionary.get_bytesList());

        dictionary.set_bytesList(markerCompressed);
        assertMatEqual(markerCompressed, dictionary.get_bytesList());
    }

    public void testArucoDetector() {
        Dictionary dictionary = Objdetect.getPredefinedDictionary(0);
        DetectorParameters detectorParameters = new DetectorParameters();
        ArucoDetector detector = new ArucoDetector(dictionary, detectorParameters);

        Mat markerImage = new Mat();
        int id = 1, offset = 5, size = 40;
        Objdetect.generateImageMarker(dictionary, id, size, markerImage, detectorParameters.get_markerBorderBits());

        Mat image = new Mat(markerImage.rows() + 2*offset, markerImage.cols() + 2*offset,
                            CvType.CV_8UC1, new Scalar(255));
        Mat m = image.submat(offset, size+offset, offset, size+offset);
        markerImage.copyTo(m);

        List<Mat> corners = new ArrayList();
        Mat ids = new Mat();
        detector.detectMarkers(image, corners, ids);

        assertEquals(1, corners.size());
        Mat res = corners.get(0);
        assertArrayEquals(new double[]{offset, offset}, res.get(0, 0), 0.0);
        assertArrayEquals(new double[]{size + offset - 1, offset}, res.get(0, 1), 0.0);
        assertArrayEquals(new double[]{size + offset - 1, size + offset - 1}, res.get(0, 2), 0.0);
        assertArrayEquals(new double[]{offset, size + offset - 1}, res.get(0, 3), 0.0);
    }

    public void testCharucoDetector() {
        Dictionary dictionary = Objdetect.getPredefinedDictionary(0);
        int boardSizeX = 3, boardSizeY = 3;
        CharucoBoard board = new CharucoBoard(new Size(boardSizeX, boardSizeY), 1.f, 0.8f, dictionary);
        CharucoDetector charucoDetector = new CharucoDetector(board);

        int cellSize = 80;
        Mat boardImage = new Mat();
        board.generateImage(new Size(cellSize*boardSizeX, cellSize*boardSizeY), boardImage);

        assertTrue(boardImage.total() > 0);

        Mat charucoCorners = new Mat();
        Mat charucoIds = new Mat();
        charucoDetector.detectBoard(boardImage, charucoCorners, charucoIds);

        assertEquals(4, charucoIds.total());
        int[] intCharucoIds = (new MatOfInt(charucoIds)).toArray();
        Assert.assertArrayEquals(new int[]{0, 1, 2, 3}, intCharucoIds);

        double eps = 0.2;
        assertArrayEquals(new double[]{cellSize, cellSize}, charucoCorners.get(0, 0), eps);
        assertArrayEquals(new double[]{2*cellSize, cellSize}, charucoCorners.get(1, 0), eps);
        assertArrayEquals(new double[]{cellSize, 2*cellSize}, charucoCorners.get(2, 0), eps);
        assertArrayEquals(new double[]{2*cellSize, 2*cellSize}, charucoCorners.get(3, 0), eps);
    }

}
