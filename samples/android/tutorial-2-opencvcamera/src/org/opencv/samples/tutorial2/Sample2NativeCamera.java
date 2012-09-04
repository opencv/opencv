package org.opencv.samples.tutorial2;

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
import android.view.WindowManager;

public class Sample2NativeCamera extends Activity {
    private static final String TAG             = "OCVSample::Activity";

    public static final int     VIEW_MODE_RGBA  = 0;
    public static final int     VIEW_MODE_GRAY  = 1;
    public static final int     VIEW_MODE_CANNY = 2;

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewGray;
    private MenuItem            mItemPreviewCanny;
    private Sample2View         mView;

    public static int           viewMode        = VIEW_MODE_RGBA;

    private BaseLoaderCallback  mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Create and set View
                    mView = new Sample2View(mAppContext);
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
                    MarketErrorMessage.setMessage("Google Play service is not accessible!\nTry to install the 'OpenCV Manager' and the appropriate 'OpenCV binary pack' APKs from OpenCV SDK manually via 'adb install' command.");
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

    public Sample2NativeCamera() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onPause() {
        Log.i(TAG, "called onPause");
        if (null != mView)
            mView.releaseCamera();
        super.onPause();
    }

    @Override
    protected void onResume() {
        Log.i(TAG, "called onResume");
        super.onResume();

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mOpenCVCallBack)) {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        }
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        else if (item == mItemPreviewGray)
            viewMode = VIEW_MODE_GRAY;
        else if (item == mItemPreviewCanny)
            viewMode = VIEW_MODE_CANNY;
        return true;
    }
}
