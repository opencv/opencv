package org.opencv.samples.s2;

import android.app.Activity;
import android.os.Bundle;
import android.view.Window;

public class Sample2Native extends Activity {
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(new Sample2View(this));
    }
}
