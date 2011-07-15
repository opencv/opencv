package org.opencv.samples.s4;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class Sample4Mixed extends Activity {
    private static final String TAG                = "Sample::Activity";

    public static final int     VIEW_MODE_RGBA     = 0;
    public static final int     VIEW_MODE_GRAY     = 1;
    public static final int     VIEW_MODE_CANNY    = 2;
    public static final int     VIEW_MODE_SOBEL    = 3;
    public static final int     VIEW_MODE_BLUR     = 4;
    public static final int     VIEW_MODE_FEATURES = 5;

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewGray;
    private MenuItem            mItemPreviewCanny;
    private MenuItem            mItemPreviewSobel;
    private MenuItem            mItemPreviewBlur;
    private MenuItem            mItemPreviewFeatures;

    public static int           viewMode           = VIEW_MODE_RGBA;

    public Sample4Mixed() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(new Sample4View(this));
    }

    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewSobel = menu.add("Sobel");
        mItemPreviewBlur = menu.add("Blur");
        mItemPreviewFeatures = menu.add("Find features");
        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        else if (item == mItemPreviewGray)
            viewMode = VIEW_MODE_GRAY;
        else if (item == mItemPreviewCanny)
            viewMode = VIEW_MODE_CANNY;
        else if (item == mItemPreviewSobel)
            viewMode = VIEW_MODE_SOBEL;
        else if (item == mItemPreviewBlur)
            viewMode = VIEW_MODE_BLUR;
        else if (item == mItemPreviewFeatures)
            viewMode = VIEW_MODE_FEATURES;
        return true;
    }
}
