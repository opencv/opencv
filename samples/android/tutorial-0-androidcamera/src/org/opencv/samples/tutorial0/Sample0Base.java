package org.opencv.samples.tutorial0;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class Sample0Base extends Activity {
    private static final String TAG            = "Sample::Activity";

    private MenuItem            mItemPreviewRGBA;
    private MenuItem            mItemPreviewGray;
    private Sample0View mView;


    public Sample0Base() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
	protected void onPause() {
        Log.i(TAG, "onPause");
		super.onPause();
		mView.releaseCamera();
	}

	@Override
	protected void onResume() {
        Log.i(TAG, "onResume");
		super.onResume();
		if( !mView.openCamera() ) {
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
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        mView = new Sample0View(this);
        setContentView(mView);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemPreviewRGBA)
        	mView.setViewMode(Sample0View.VIEW_MODE_RGBA);
        else if (item == mItemPreviewGray)
        	mView.setViewMode(Sample0View.VIEW_MODE_GRAY);
        return true;
    }
}
