package org.opencv.samples.colorblobdetect;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.View.OnTouchListener;

public class ColorBlobDetectionView extends SampleCvViewBase implements OnTouchListener {

	private Mat mRgba;

	private boolean mIsColorSelected = false;
	private Scalar mSelectedColorRgba = new Scalar(255);
	private Scalar mSelectedColorHsv = new Scalar(255);
	
	// Lower and Upper bounds for range checking in HSV color space
	private Scalar mLowerBound = new Scalar(0);
	private Scalar mUpperBound = new Scalar(0);
	
	private Mat mSpectrum = new Mat();
	private int mSpectrumScale = 4;

	// Color radius for range checking in HSV color space
	private static final Scalar COLOR_RADIUS = new Scalar(25,50,50,0);

	// Minimum contour area in percent for contours filtering
	private static final double MIN_CONTOUR_AREA = 0.1;

	// Logcat tag
	private static final String TAG = "Example/CollorBlobDetection";
	
	
	public ColorBlobDetectionView(Context context)
	{
        super(context);
        setOnTouchListener(this);
	}
	
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mat before usage
            mRgba = new Mat();
        }
        
        super.surfaceCreated(holder);
    }
	
	public boolean onTouch(View v, MotionEvent event)
	{
        int cols = mRgba.cols();
        int rows = mRgba.rows();
        
        int xOffset = (getWidth() - cols) / 2;
        int yOffset = (getHeight() - rows) / 2;
        
        int x = (int)event.getX() - xOffset;
        int y = (int)event.getY() - yOffset;
        
        Log.i(TAG, "Touch image coordinates: (" + x + ", " + y + ")");
        
        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;
  
        Rect touchedRect = new Rect();
        
        touchedRect.x = (x>4) ? x-4 : 0;
        touchedRect.y = (y>4) ? y-4 : 0;

        touchedRect.width = (x+4<mRgba.cols()) ? x + 4 - touchedRect.x : mRgba.width() - touchedRect.x;
        touchedRect.height = (y+4 < mRgba.rows()) ? y + 4 - touchedRect.y : mRgba.rows() - touchedRect.y;
        	
        Mat touchedRegionMatRgba = mRgba.submat(touchedRect);
        Mat touchedRegionMatHsv = new Mat();
        
        Imgproc.cvtColor(touchedRegionMatRgba, touchedRegionMatHsv, Imgproc.COLOR_RGB2HSV_FULL);
        
        mSelectedColorHsv = Core.sumElems(touchedRegionMatHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mSelectedColorHsv.val.length; i++)
        {
        	mSelectedColorHsv.val[i] /= pointCount;
        }
        
        Mat pointMapRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3);
        
        byte[] buf = {(byte)mSelectedColorHsv.val[0], (byte)mSelectedColorHsv.val[1], (byte)mSelectedColorHsv.val[2]};
        
        pointMatHsv.put(0, 0, buf);
        Imgproc.cvtColor(pointMatHsv, pointMapRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);
        
        mSelectedColorRgba.val = pointMapRgba.get(0, 0);
        
        Log.i(TAG, "Touched rgba color: (" + mSelectedColorRgba.val[0] + ", " + mSelectedColorRgba.val[1] + 
    			", " + mSelectedColorRgba.val[2] + ", " + mSelectedColorRgba.val[3] + ")");
        
    	double minH = (mSelectedColorHsv.val[0] >= COLOR_RADIUS.val[0]) ? mSelectedColorHsv.val[0]-COLOR_RADIUS.val[0] : 0; 
    	double maxH = (mSelectedColorHsv.val[0]+COLOR_RADIUS.val[0] <= 255) ? mSelectedColorHsv.val[0]+COLOR_RADIUS.val[0] : 255;
            	
  		mLowerBound.val[0] = minH;
   		mUpperBound.val[0] = maxH;
   		
  		mLowerBound.val[1] = mSelectedColorHsv.val[1] - COLOR_RADIUS.val[1];
   		mUpperBound.val[1] = mSelectedColorHsv.val[1] + COLOR_RADIUS.val[1];
   		
  		mLowerBound.val[2] = mSelectedColorHsv.val[2] - COLOR_RADIUS.val[2];
   		mUpperBound.val[2] = mSelectedColorHsv.val[2] + COLOR_RADIUS.val[2];
   		
    	Log.d(TAG, "Bounds: " + mLowerBound + "x" + mUpperBound);
   		
   		Mat spectrumHsv = new Mat(32, (int)(maxH-minH)*mSpectrumScale, CvType.CV_8UC3);
   		
   		for (int i = 0; i < 32; i++)
   		{
   			for (int k = 0; k < mSpectrumScale; k++)
   			{
   				for (int j = 0; j < maxH-minH; j++)
   				{
   					byte[] tmp = {(byte)(minH+j), (byte)255, (byte)255};
   					spectrumHsv.put(i, j*mSpectrumScale + k, tmp);
   				}
   			}
   		}
        
   		Imgproc.cvtColor(spectrumHsv, mSpectrum, Imgproc.COLOR_HSV2RGB_FULL, 4);
   		
        mIsColorSelected = true;
        
        return false; // don't need subsequent touch events
	}

	@Override
	protected Bitmap processFrame(VideoCapture capture) {
		capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
		
        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
        
        if (mIsColorSelected)
        {        	
        	Mat PyrDownMat = new Mat();
        	
        	Imgproc.pyrDown(mRgba, PyrDownMat);
        	Imgproc.pyrDown(PyrDownMat, PyrDownMat);
        	
        	Mat hsvMat = new Mat();
        	Imgproc.cvtColor(PyrDownMat, hsvMat, Imgproc.COLOR_RGB2HSV_FULL);
        	
        	Mat rangedHsvMat = new Mat();
        	Core.inRange(hsvMat, mLowerBound, mUpperBound, rangedHsvMat);
        	
        	Mat dilatedMat = new Mat();
        	Imgproc.dilate(rangedHsvMat, dilatedMat, new Mat());
        	        	
            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            Mat hierarchy = new Mat();

            Imgproc.findContours(dilatedMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            
            // Find max contour area
            double maxArea = 0;
            Iterator<MatOfPoint> it = contours.iterator();
            while (it.hasNext())
            {
            	MatOfPoint wrapper = it.next();
            	double area = Imgproc.contourArea(wrapper);
            	if (area > maxArea)
            		maxArea = area;
            }
            
            // Filter contours by area and resize to fit the original image size
            List<MatOfPoint> filteredContours = new ArrayList<MatOfPoint>();
            it = contours.iterator();
            while (it.hasNext())
            {
            	MatOfPoint wrapper = it.next();
            	if (Imgproc.contourArea(wrapper) > MIN_CONTOUR_AREA*maxArea);
            	Point[] contour = wrapper.toArray();
            	for (int i = 0; i < contour.length; i++)
            	{
            		// Original image was pyrDown twice
            		contour[i].x *= 4;
            		contour[i].y *= 4;
            	}
            	filteredContours.add(new MatOfPoint(contour));
            }
            
            Core.drawContours(mRgba, filteredContours, -1, new Scalar(255,0,0,255));
            
            Mat testColorMat = mRgba.submat(2, 34, 2, 34);
            testColorMat.setTo(mSelectedColorRgba);
            
            Mat testSpectrumMat = mRgba.submat(2, 34, 38, 38 + mSpectrum.cols());
            mSpectrum.copyTo(testSpectrumMat);
        }

        try {
        	Utils.matToBitmap(mRgba, bmp);
        } catch(Exception e) {
        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }
        
        return bmp;
	}
	
    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();

            mRgba = null;
        }
    }
}
