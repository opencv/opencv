package org.opencv.samples;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class Sample0Base extends Activity {
    private static final String TAG = "Sample0Base::Activity";
    
	public static final int view_mode_rgba = 0;
	public static final int view_mode_gray = 1;
	
	private MenuItem item_preview_rgba;
	private MenuItem item_preview_gray;
	
	public int view_mode;
	
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView( new Sample0View(this) );
        view_mode = view_mode_rgba;
    }
    
    public boolean onCreateOptionsMenu(Menu menu) {
    	item_preview_rgba = menu.add("Preview RGBA");
    	item_preview_gray = menu.add("Preview GRAY");
		return true;
    }
    
    public boolean onOptionsItemSelected (MenuItem item) {
    	Log.i(TAG, "Menu Item selected " + item);
    	if (item == item_preview_rgba)
    		view_mode = view_mode_rgba;
    	else if (item == item_preview_gray)
    		view_mode = view_mode_gray;
		return true;
    }
}
