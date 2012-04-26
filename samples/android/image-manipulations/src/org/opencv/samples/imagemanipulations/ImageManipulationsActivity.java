package org.opencv.samples.imagemanipulations;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class ImageManipulationsActivity extends Activity {
    private static final String TAG             = "Sample::Activity";

    public static final int     VIEW_MODE_RGBA  = 0;
    public static final int     VIEW_MODE_CANNY = 1;
    public static final int     VIEW_MODE_SEPIA = 2;
    public static final int     VIEW_MODE_SOBEL = 3;
    public static final int     VIEW_MODE_BLUR  = 4;
    public static final int     VIEW_MODE_ZOOM  = 5;
    public static final int     VIEW_MODE_PIXELIZE  = 6;

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewCanny;
    private MenuItem            mItemPreviewSepia;
    private MenuItem            mItemPreviewSobel;
    private MenuItem            mItemPreviewBlur;
    private MenuItem            mItemPreviewZoom;
    private MenuItem            mItemPreviewPixelize;

    public static int           viewMode = VIEW_MODE_RGBA;

    public ImageManipulationsActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(new ImageManipulationsView(this));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewSepia = menu.add("Sepia");
        mItemPreviewSobel = menu.add("Sobel");
        mItemPreviewBlur = menu.add("Blur");
        mItemPreviewZoom = menu.add("Zoom");
        mItemPreviewPixelize = menu.add("Pixelize");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        else if (item == mItemPreviewCanny)
            viewMode = VIEW_MODE_CANNY;
        else if (item == mItemPreviewSepia)
            viewMode = VIEW_MODE_SEPIA;
        else if (item == mItemPreviewSobel)
            viewMode = VIEW_MODE_SOBEL;
        else if (item == mItemPreviewBlur)
            viewMode = VIEW_MODE_BLUR;
        else if (item == mItemPreviewZoom)
            viewMode = VIEW_MODE_ZOOM;
        else if (item == mItemPreviewPixelize)
            viewMode = VIEW_MODE_PIXELIZE;
        return true;
    }
}
