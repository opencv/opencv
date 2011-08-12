package org.opencv.samples.puzzle15;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;

public class puzzle15Activity extends Activity {
    private static final String TAG = "Sample::Activity";

    private MenuItem            mItemNewGame;
    private MenuItem            mItemToggleNumbers;
    private puzzle15View        mView;

    public puzzle15Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        mView = new puzzle15View(this);
        setContentView(mView);
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