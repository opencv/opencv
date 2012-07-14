package org.opencv.samples.fd;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class FdActivity extends Activity {
    private static final String TAG         = "Sample::Activity";

    private MenuItem            mItemFace50;
    private MenuItem            mItemFace40;
    private MenuItem            mItemFace30;
    private MenuItem            mItemFace20;
    private MenuItem            mItemType;
    
    private FdView		mView;
    
    private BaseLoaderCallback  mOpenCVCallBack = new BaseLoaderCallback(this) {
    	@Override
    	public void onManagerConnected(int status) {
    		switch (status) {
				case LoaderCallbackInterface.SUCCESS:
				{
					Log.i(TAG, "OpenCV loaded successfully");
					
					// Load native libs after OpenCV initialization
					System.loadLibrary("detection_based_tracker");

					// Create and set View
					mView = new FdView(mAppContext);
					mView.setDetectorType(mDetectorType);
					mView.setMinFaceSize(0.2f);
					setContentView(mView);
					// Check native OpenCV camera
					if( !mView.openCamera() ) {
						AlertDialog ad = new AlertDialog.Builder(mAppContext).create();
						ad.setCancelable(false); // This blocks the 'BACK' button
						ad.setMessage("Fatal error: can't open camera!");
						ad.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new DialogInterface.OnClickListener() {
						    public void onClick(DialogInterface dialog, int which) {
							dialog.dismiss();
							finish();
						    }
						});
						ad.show();
					}
				} break;
				default:
				{
					super.onManagerConnected(status);
				} break;
			}
    	}
    };


    private int                 mDetectorType = 0;
    private String[]            mDetectorName; 

    public FdActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
        mDetectorName = new String[2];
        mDetectorName[FdView.JAVA_DETECTOR] = "Java";
        mDetectorName[FdView.NATIVE_DETECTOR] = "Native (tracking)";
    }

	@Override
	protected void onPause() {
        Log.i(TAG, "onPause");
		super.onPause();
		if (mView != null)
			mView.releaseCamera();
	}

	@Override
	protected void onResume() {
        Log.i(TAG, "onResume");
		super.onResume();
		if( mView != null && !mView.openCamera() ) {
			AlertDialog ad = new AlertDialog.Builder(this).create();  
			ad.setCancelable(false); // This blocks the 'BACK' button  
			ad.setMessage("Fatal error: can't open camera!");  
			ad.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new DialogInterface.OnClickListener() {  
			    public void onClick(DialogInterface dialog, int which) {  
			        dialog.dismiss();                      
					finish();
			    }  
			});  
			ad.show();
		}
	}

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mOpenCVCallBack))
        {
        	Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        		
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemFace50)
            mView.setMinFaceSize(0.5f);
        else if (item == mItemFace40)
        	mView.setMinFaceSize(0.4f);
        else if (item == mItemFace30)
        	mView.setMinFaceSize(0.3f);
        else if (item == mItemFace20)
        	mView.setMinFaceSize(0.2f);
        else if (item == mItemType)
        {
        	mDetectorType = (mDetectorType + 1) % mDetectorName.length;
        	item.setTitle(mDetectorName[mDetectorType]);
        	mView.setDetectorType(mDetectorType);
        }
        return true;
    }
}
