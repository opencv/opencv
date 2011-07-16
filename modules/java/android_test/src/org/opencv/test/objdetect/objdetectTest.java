package org.opencv.test.objdetect;

import java.util.ArrayList;

import org.opencv.Mat;
import org.opencv.objdetect;
import org.opencv.imgproc;
import org.opencv.highgui;
import org.opencv.core;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.Rect;
import org.opencv.Size;

public class objdetectTest extends OpenCVTestCase {
	public void testCascadeClassifierFaceDetector() {
		//objdetect.CascadeClassifier cc=new objdetect.CascadeClassifier("/mnt/sdcard/lbpcascade_frontalface.xml");
		objdetect.CascadeClassifier cc=new objdetect.CascadeClassifier("/mnt/sdcard/haarcascade_frontalface_alt2.xml");
		ArrayList<Rect> faces=new ArrayList<Rect>();
		
		
		Mat shot002=highgui.imread("/mnt/sdcard/shot0002.png");
		OpenCVTestRunner.Log("after imread shot002");
		
		cc.detectMultiScale(shot002, faces, 1.1, 2, 2 /*TODO: CV_HAAR_SCALE_IMAGE*/, new Size(10,10));
		OpenCVTestRunner.Log("faces.size="+faces.size());
	}
	

}
