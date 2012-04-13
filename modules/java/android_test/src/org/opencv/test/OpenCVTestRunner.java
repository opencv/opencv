package org.opencv.test;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import junit.framework.TestCase;
import junit.framework.Assert;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import android.content.Context;
import android.test.AndroidTestRunner;
import android.test.InstrumentationTestRunner;
import android.util.Log;

/**
 * This only class is Android specific.
 * 
 * @see <a href="http://opencv.itseez.com">OpenCV</a>
 */

public class OpenCVTestRunner extends InstrumentationTestRunner {

    public static String LENA_PATH;
    public static String CHESS_PATH;
    public static String LBPCASCADE_FRONTALFACE_PATH;
    public static Context context;

    private AndroidTestRunner androidTestRunner;
    private static String TAG = "opencv_test_java";

    public static String getTempFileName(String extension)
    {
        File cache = context.getCacheDir();
        if (!extension.startsWith("."))
            extension = "." + extension;
        try {
            File tmp = File.createTempFile("OpenCV", extension, cache);
            String path = tmp.getAbsolutePath();
            tmp.delete();
            return path;
        } catch (IOException e) {
            Log("Failed to get temp file name. Exception is thrown: " + e);
        }
        return null;
    }

    static public void Log(String message) {
        Log.e(TAG, message);
    }

    static public void Log(Mat m) {
        Log.e(TAG, m + "\n " + m.dump());
    }

    @Override
    public void onStart() {
        context = getContext();
        Assert.assertTrue("Context can't be 'null'", context != null);
        LENA_PATH = Utils.exportResource(context, R.drawable.lena);
        CHESS_PATH = Utils.exportResource(context, R.drawable.chessboard);
        LBPCASCADE_FRONTALFACE_PATH = Utils.exportResource(context, R.raw.lbpcascade_frontalface);

        /*
         * The original idea about test order randomization is from
         * marek.defecinski blog.
         */
        //List<TestCase> testCases = androidTestRunner.getTestCases();
        //Collections.shuffle(testCases); //shuffle the tests order

        if(OpenCVTestCase.passNYI) {
            // turn off problematic camera tests
            Iterator<TestCase> it = androidTestRunner.getTestCases().iterator();
            while (it.hasNext()) {
                String name = it.next().toString();
                if (name.contains("VideoCaptureTest"))
                    it.remove();
            }
        }


        super.onStart();
    }

    @Override
    protected AndroidTestRunner getAndroidTestRunner() {
        androidTestRunner = super.getAndroidTestRunner();
        return androidTestRunner;
    }

    public static String getOutputFileName(String name)
    {
        return context.getExternalFilesDir(null).getAbsolutePath() + File.separatorChar + name;
    }
}
