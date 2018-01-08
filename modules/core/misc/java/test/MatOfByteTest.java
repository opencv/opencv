package org.opencv.test.core;

import java.util.Arrays;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.io.File;
import java.io.IOException;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgcodecs.Imgcodecs;

public class MatOfByteTest extends OpenCVTestCase {
    public static String LENA_W_10BYTEMARGIN_PATH = "";
    public static String CHESS_W_10BYTEMARGIN_PATH = "";
    Mat rgbLena2;
    Mat grayChess2;
    Mat rgbLenawMargin;
    Mat grayChesswMargin;
    
    @Override
    protected void setUp() throws Exception {
        super.setUp();
        LENA_W_10BYTEMARGIN_PATH = OpenCVTestRunner.LENA_PATH.replace("lena.png", "lena-10byte_margin.png");
        CHESS_W_10BYTEMARGIN_PATH = OpenCVTestRunner.CHESS_PATH.replace("chessboard.jpg", "chessboard-10byte_margin.jpg");
        Path pathOfLena = Paths.get(OpenCVTestRunner.LENA_PATH);
        Path pathOfChess = Paths.get(OpenCVTestRunner.CHESS_PATH);
        Path pathOfLenawMargin = Paths.get(LENA_W_10BYTEMARGIN_PATH);
        Path pathOfChesswMargin = Paths.get(CHESS_W_10BYTEMARGIN_PATH);
        
        byte[] byteArrayLena = Files.readAllBytes(pathOfLena);
        byte[] byteArrayChess = Files.readAllBytes(pathOfChess);
        byte[] byteArrayLenawMargin = Files.readAllBytes(pathOfLenawMargin);
        byte[] byteArrayChesswMargin = Files.readAllBytes(pathOfChesswMargin);
        
        MatOfByte matLena = new MatOfByte(0, byteArrayLena.length, byteArrayLena);
        MatOfByte matChess = new MatOfByte(0, byteArrayChess.length, byteArrayChess);
        MatOfByte matLenawMargin = new MatOfByte(10, byteArrayLenawMargin.length - 20, byteArrayLenawMargin);
        MatOfByte matChesswMargin = new MatOfByte(10, byteArrayChesswMargin.length - 20, byteArrayChesswMargin);
        
        rgbLena2 = Imgcodecs.imdecode(matLena, Imgcodecs.IMREAD_COLOR);;
        grayChess2 = Imgcodecs.imdecode(matChess, Imgcodecs.IMREAD_GRAYSCALE);
        rgbLenawMargin = Imgcodecs.imdecode(matLenawMargin, Imgcodecs.IMREAD_COLOR);
        grayChesswMargin = Imgcodecs.imdecode(matChesswMargin, Imgcodecs.IMREAD_GRAYSCALE);
    }
    
    public void testMatOfSubByteArray() {
        assertEquals(3, rgbLenawMargin.channels());
        assertEquals(512, rgbLenawMargin.cols());
        assertEquals(512, rgbLenawMargin.rows());
        assertEquals(512, rgbLenawMargin.width());
        assertEquals(512, rgbLenawMargin.height());
        
        assertEquals(3, rgbLena2.channels());
        assertEquals(512, rgbLena2.cols());
        assertEquals(512, rgbLena2.rows());
        assertEquals(512, rgbLena2.width());
        assertEquals(512, rgbLena2.height());
        
        MatOfDouble mean   = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();

        Core.meanStdDev(rgbLena2, mean, stddev);
        double expectedMean[] = new double[]
            {105.3989906311035, 99.56269836425781, 179.7303047180176};
        double expectedDev[] = new double[]
            {33.74205485167219, 52.8734582803278, 49.01569488056406};
        assertArrayEquals(expectedMean, mean.toArray(), EPS);
        assertArrayEquals(expectedDev, stddev.toArray(), EPS);
        
        Core.meanStdDev(rgbLenawMargin, mean, stddev);
        assertArrayEquals(expectedMean, mean.toArray(), EPS);
        assertArrayEquals(expectedDev, stddev.toArray(), EPS);
        
        assertEquals(1, grayChess2.channels());
        assertEquals(640, grayChess2.cols());
        assertEquals(480, grayChess2.rows());
        assertEquals(640, grayChess2.width());
        assertEquals(480, grayChess2.height());
        assertEquals(1, grayChesswMargin.channels());
        assertEquals(640, grayChesswMargin.cols());
        assertEquals(480, grayChesswMargin.rows());
        assertEquals(640, grayChesswMargin.width());
        assertEquals(480, grayChesswMargin.height());
    }

}
