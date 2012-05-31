package org.opencv.samples.colorblobdetect;

import java.util.List;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
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
	private Scalar mBlobColorRgba = new Scalar(255);
	private Scalar mBlobColorHsv = new Scalar(255);
	private ColorBlobDetector mDetector = new ColorBlobDetector();
	private Mat mSpectrum = new Mat();
	private static Size SPECTRUM_SIZE = new Size(200, 32);

	// Logcat tag
	private static final String TAG = "Example/ColorBlobDetection";
	
	private static final Scalar CONTOUR_COLOR = new Scalar(255,0,0,255);
	
	
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

        touchedRect.width = (x+4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y+4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;
        	
        Mat touchedRegionRgba = mRgba.submat(touchedRect);
        
        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);
        
        // Calculate average color of touched region
        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
        {
        	mBlobColorHsv.val[i] /= pointCount;
        }
        
        mBlobColorRgba = converScalarHsv2Rgba(mBlobColorHsv);
        
        Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba.val[0] + ", " + mBlobColorRgba.val[1] + 
    			", " + mBlobColorRgba.val[2] + ", " + mBlobColorRgba.val[3] + ")");
   		
   		mDetector.setHsvColor(mBlobColorHsv);
   		
   		Imgproc.resize(mDetector.getSpectrum(), mSpectrum, SPECTRUM_SIZE);
   		
        mIsColorSelected = true;
        
        return false; // don't need subsequent touch events
	}

	@Override
	protected Bitmap processFrame(VideoCapture capture) {
		capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
		
        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
        
        if (mIsColorSelected)
        {            
        	mDetector.process(mRgba);
        	List<MatOfPoint> contours = mDetector.getContours();
            Log.e(TAG, "Contours count: " + contours.size());
        	Core.drawContours(mRgba, contours, -1, CONTOUR_COLOR);
            
            Mat colorLabel = mRgba.submat(2, 34, 2, 34);
            colorLabel.setTo(mBlobColorRgba);
            
            Mat spectrumLabel = mRgba.submat(2, 2 + mSpectrum.rows(), 38, 38 + mSpectrum.cols());
            mSpectrum.copyTo(spectrumLabel);
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
	
	private Scalar converScalarHsv2Rgba(Scalar hsvColor)
	{	
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);
        
        return new Scalar(pointMatRgba.get(0, 0));
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
