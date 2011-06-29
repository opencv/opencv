package org.opencv.samples;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class Sample0Base extends Activity {
    private static final String TAG = "Sample0Base::Activity";

    public static final int VIEW_MODE_RGBA = 0;
    public static final int VIEW_MODE_GRAY = 1;

    private MenuItem mItemPreviewRGBA;
    private MenuItem mItemPreviewGray;

    public int viewMode;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(new Sample0View(this));
        viewMode = VIEW_MODE_RGBA;
    }

    public boolean onCreateOptionsMenu(Menu menu) {
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        else if (item == mItemPreviewGray)
            viewMode = VIEW_MODE_GRAY;
        return true;
    }
}
