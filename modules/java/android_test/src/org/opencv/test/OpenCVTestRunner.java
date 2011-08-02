package org.opencv.test;

import android.content.Context;
import android.test.AndroidTestRunner;
import android.test.InstrumentationTestRunner;
import android.util.Log;

import org.opencv.Android;

import java.io.File;
import java.io.IOException;

/**
 * This only class is Android specific. The original idea about test order
 * randomization is from marek.defecinski blog.
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

    @Override
    public void onStart() {
        context = getContext();
        LENA_PATH = Android.ExportResource(context, R.drawable.lena);
        CHESS_PATH = Android.ExportResource(context, R.drawable.chessboard);
        LBPCASCADE_FRONTALFACE_PATH = Android.ExportResource(context, R.raw.lbpcascade_frontalface);

        // List<TestCase> testCases = androidTestRunner.getTestCases();
        // Collections.shuffle(testCases); //shuffle the tests order

        super.onStart();
    }

    @Override
    protected AndroidTestRunner getAndroidTestRunner() {
        androidTestRunner = super.getAndroidTestRunner();
        return androidTestRunner;
    }
}
