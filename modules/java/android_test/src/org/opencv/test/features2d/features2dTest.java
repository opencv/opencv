package org.opencv.test.features2d;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features2dTest extends OpenCVTestCase {
    
    public void testPTOD()
    {
        String detectorCfg = "%YAML:1.0\nhessianThreshold: 4000.\noctaves: 3\noctaveLayers: 4\nupright: 0\n";
        String extractorCfg = "%YAML:1.0\nnOctaves: 4\nnOctaveLayers: 2\nextended: 0\nupright: 0\n";
        
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        
        String detectorCfgFile = OpenCVTestRunner.getTempFileName("yml");
        writeFile(detectorCfgFile, detectorCfg);
        detector.read(detectorCfgFile);
        
        String extractorCfgFile = OpenCVTestRunner.getTempFileName("yml");
        writeFile(extractorCfgFile, extractorCfg);
        extractor.read(extractorCfgFile);
        
        Mat imgTrain = Highgui.imread(OpenCVTestRunner.LENA_PATH, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        Mat imgQuery = imgTrain.submat(new Range(0, imgTrain.rows()-100), Range.all());
        
        List<KeyPoint> trainKeypoints = new ArrayList<KeyPoint>();
        List<KeyPoint> queryKeypoints = new ArrayList<KeyPoint>();
        
        detector.detect(imgTrain, trainKeypoints);
        detector.detect(imgQuery, queryKeypoints);
        
        //OpenCVTestRunner.Log("Keypoints found: " + trainKeypoints.size() + ":" + queryKeypoints.size());
        
        Mat trainDescriptors = new Mat();
        Mat queryDescriptors = new Mat();
        
        extractor.compute(imgTrain, trainKeypoints, trainDescriptors);
        extractor.compute(imgQuery, queryKeypoints, queryDescriptors);
        
        List<DMatch> matches = new ArrayList<DMatch>();
        
        matcher.add(Arrays.asList(trainDescriptors));
        matcher.match(queryDescriptors, matches);
        
        //OpenCVTestRunner.Log("Matches found: " + matches.size());
        
        List<Point> points1 = new ArrayList<Point>();
        List<Point> points2 = new ArrayList<Point>();
        
        for(int i = 0; i < matches.size(); i++){
            DMatch match = matches.get(i);
            points1.add(trainKeypoints.get(match.trainIdx).pt);
            points2.add(queryKeypoints.get(match.queryIdx).pt);
        }
        
        Mat hmg = Calib3d.findHomography(points1, points2, Calib3d.RANSAC);
        
        assertMatEqual(Mat.eye(3, 3, CvType.CV_64F), hmg, EPS);
        
        Mat outimg = new Mat();
        Features2d.drawMatches(imgQuery, queryKeypoints, imgTrain, trainKeypoints, matches, outimg);
        String outputPath = OpenCVTestRunner.getOutputFileName("PTODresult.png");
        Highgui.imwrite(outputPath, outimg);
        //OpenCVTestRunner.Log("Output image is saved to: " + outputPath);
    }
}
