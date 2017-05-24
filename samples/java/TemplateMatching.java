package com.mesutpiskin.templatematcing;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
/*
Template Matching Method for Object Detection
https://camo.githubusercontent.com/6cb87bdd27c172129e74750d00192e809540c20b/687474703a2f2f692e737461636b2e696d6775722e636f6d2f4a496f51382e6a7067

*/
public class TemplateMatching {

	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat source=null;
		Mat template=null;
		String filePath="Sample Image\\";
		source=Imgcodecs.imread(filePath+"kapadokya.jpg");
		template=Imgcodecs.imread(filePath+"balon.jpg");
	
		Mat outputImage=new Mat();	
		int machMethod=Imgproc.TM_CCOEFF;
   
       		Imgproc.matchTemplate(source, template, outputImage, machMethod);
 
    
       		MinMaxLocResult mmr = Core.minMaxLoc(outputImage);
      		Point matchLoc=mmr.maxLoc;

       	             Imgproc.rectangle(source, matchLoc, new Point(matchLoc.x + template.cols(),
                matchLoc.y + template.rows()), new Scalar(255, 255, 255));

        	             Imgcodecs.imwrite(filePath+"sonuc.jpg", source);
                          System.out.println("Done.");
	}

}
