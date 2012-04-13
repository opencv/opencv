package org.opencv.samples.tutorial4;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class Sample4Mixed extends Activity {
    private static final String TAG = "Sample::Activity";

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewGray;
    private MenuItem            mItemPreviewCanny;
    private MenuItem            mItemPreviewFeatures;
    private Sample4View         mView;    


    public Sample4Mixed() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        mView = new Sample4View(this);
        setContentView(mView);
    }

    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewFeatures = menu.add("Find features");
        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA) {
        	mView.setViewMode(Sample4View.VIEW_MODE_RGBA);
        } else if (item == mItemPreviewGray) {
        	mView.setViewMode(Sample4View.VIEW_MODE_GRAY);
        } else if (item == mItemPreviewCanny) {
        	mView.setViewMode(Sample4View.VIEW_MODE_CANNY);
        } else if (item == mItemPreviewFeatures) {
        	mView.setViewMode(Sample4View.VIEW_MODE_FEATURES);
        }
        return true;
    }
}
