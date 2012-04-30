package org.opencv.samples.colorblobdetect;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;

public class ColorBlobDetectionActivity extends Activity {
	
	private static final String TAG = "Example/CollorBlobDetection";
	private ColorBlobDetectionView mView;
	
	public ColorBlobDetectionActivity()
	{
		Log.i(TAG, "Instantiated new " + this.getClass());
	}
	
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {        
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        mView = new ColorBlobDetectionView(this);
        setContentView(mView);
    }
}