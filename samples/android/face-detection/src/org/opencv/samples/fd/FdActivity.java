package org.opencv.samples.fd;

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
    
    private FdView				mView;
    
    private int                 mDetectorType = 0;
    private String[]            mDetectorName; 

    public FdActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
        mDetectorName = new String[2];
        mDetectorName[0] = "Cascade";
        mDetectorName[1] = "DBT";
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
        mView = new FdView(this);
        mView.setDtetectorType(mDetectorType);
        setContentView(mView);
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
        	mView.setDtetectorType(mDetectorType);
        }
        return true;
    }
}
