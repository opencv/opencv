package org.opencv.samples.imagemanipulations;

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

public class ImageManipulationsActivity extends Activity {

	private static final String TAG = "Sample-ImageManipulations::Activity";

    public static final int     VIEW_MODE_RGBA      = 0;
    public static final int     VIEW_MODE_HIST      = 1;
    public static final int     VIEW_MODE_CANNY     = 2;
    public static final int     VIEW_MODE_SEPIA     = 3;
    public static final int     VIEW_MODE_SOBEL     = 4;
    public static final int     VIEW_MODE_ZOOM      = 5;
    public static final int     VIEW_MODE_PIXELIZE  = 6;
    public static final int     VIEW_MODE_POSTERIZE = 7;

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewHist;
    private MenuItem            mItemPreviewCanny;
    private MenuItem            mItemPreviewSepia;
    private MenuItem            mItemPreviewSobel;
    private MenuItem            mItemPreviewZoom;
    private MenuItem            mItemPreviewPixelize;
    private MenuItem            mItemPreviewPosterize;

    public static int           viewMode = VIEW_MODE_RGBA;
    
    private ImageManipulationsView mView;
    
    private BaseLoaderCallback  mOpenCVCallBack = new BaseLoaderCallback(this) {
    	@Override
    	public void onManagerConnected(int status) {
    		switch (status) {
				case LoaderCallbackInterface.SUCCESS:
				{
					Log.i(TAG, "OpenCV loaded successfully");
					// Create and set View
					mView = new ImageManipulationsView(mAppContext);
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
				/** OpenCV loader cannot start Google Play **/
				case LoaderCallbackInterface.MARKET_ERROR:
				{
					Log.d(TAG, "Google Play service is not accessible!");
					AlertDialog MarketErrorMessage = new AlertDialog.Builder(mAppContext).create();
					MarketErrorMessage.setTitle("OpenCV Manager");
					MarketErrorMessage.setMessage("Google Play service is not accessible!\nTry to install the 'OpenCV Manager' and the appropriate 'OpenCV binary pack' APKs from OpenCV SDK manually via 'adb install file_name.apk' command.");
					MarketErrorMessage.setCancelable(false); // This blocks the 'BACK' button
					MarketErrorMessage.setButton(AlertDialog.BUTTON_POSITIVE, "OK", new DialogInterface.OnClickListener() {
						public void onClick(DialogInterface dialog, int which) {
							mAppContext.finish();
						}
					});
					MarketErrorMessage.show();
				} break;
				default:
				{
					super.onManagerConnected(status);
				} break;
			}
    	}
	};

    public ImageManipulationsActivity() {
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
		if( (null != mView) && !mView.openCamera() ) {
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
        mItemPreviewRGBA  = menu.add("Preview RGBA");
        mItemPreviewHist  = menu.add("Histograms");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewSepia = menu.add("Sepia");
        mItemPreviewSobel = menu.add("Sobel");
        mItemPreviewZoom  = menu.add("Zoom");
        mItemPreviewPixelize  = menu.add("Pixelize");
        mItemPreviewPosterize = menu.add("Posterize");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        if (item == mItemPreviewHist)
            viewMode = VIEW_MODE_HIST;
        else if (item == mItemPreviewCanny)
            viewMode = VIEW_MODE_CANNY;
        else if (item == mItemPreviewSepia)
            viewMode = VIEW_MODE_SEPIA;
        else if (item == mItemPreviewSobel)
            viewMode = VIEW_MODE_SOBEL;
        else if (item == mItemPreviewZoom)
            viewMode = VIEW_MODE_ZOOM;
        else if (item == mItemPreviewPixelize)
            viewMode = VIEW_MODE_PIXELIZE;
        else if (item == mItemPreviewPosterize)
            viewMode = VIEW_MODE_POSTERIZE;
        return true;
    }
}
