package org.opencv.samples.puzzle15;

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

/** Activity class implements LoaderCallbackInterface to handle OpenCV initialization status **/
public class puzzle15Activity extends Activity
{
	private static final String TAG = "Sample::Activity";

    private MenuItem            mItemNewGame;
    private MenuItem            mItemToggleNumbers;
    private puzzle15View        mView = null;
    private BaseLoaderCallback  mOpenCVCallBack = new BaseLoaderCallback(this) {
    	@Override
    	public void onManagerConnected(int status) {
    		switch (status) {
				case LoaderCallbackInterface.SUCCESS:
				{
					Log.i(TAG, "OpenCV loaded successfully");
					// Create and set View
					mView = new puzzle15View(mAppContext);
					setContentView(mView);
					// Check native OpenCV camera
					if( !mView.openCamera() ) {
						AlertDialog ad = new AlertDialog.Builder(mAppContext).create();
						ad.setCancelable(false); // This blocks the 'BACK' button
						ad.setMessage("Fatal error: can't open camera!");
						ad.setButton("OK", new DialogInterface.OnClickListener() {
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
    
    public puzzle15Activity()
    {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
	protected void onPause() {
        Log.i(TAG, "onPause");
		super.onPause();
		if (null != mView)
			mView.releaseCamera();
	}

	@Override
	protected void onResume() {
        Log.i(TAG, "onResume");
		super.onResume();
		if( mView!=null && !mView.openCamera() ) {
			AlertDialog ad = new AlertDialog.Builder(this).create();
			ad.setCancelable(false); // This blocks the 'BACK' button
			ad.setMessage("Fatal error: can't open camera!");
			ad.setButton("OK", new DialogInterface.OnClickListener() {
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
    public void onCreate(Bundle savedInstanceState)
    {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPEN_CV_VERSION_2_4_0, this, mOpenCVCallBack))
        {
        	Log.e(TAG, "Cannot connect to OpenCV Manager");
        	finish();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemNewGame = menu.add("Start new game");
        mItemToggleNumbers = menu.add("Show/hide tile numbers");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemNewGame) {
            synchronized (mView) {
                mView.startNewGame();
            }
        } else if (item == mItemToggleNumbers)
            mView.tolggleTileNumbers();
        return true;
    }
}