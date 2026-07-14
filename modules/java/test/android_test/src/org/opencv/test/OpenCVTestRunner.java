package org.opencv.test;

import java.io.File;
import java.io.IOException;
import junit.framework.Assert;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import android.content.Context;
import android.util.Log;

import androidx.test.runner.AndroidJUnitRunner;


/**
 * This only class is Android specific.
 */

public class OpenCVTestRunner extends AndroidJUnitRunner {

    private static final long MANAGER_TIMEOUT = 3000;
    public static String LENA_PATH;
    public static String CHESS_PATH;
    public static String LBPCASCADE_FRONTALFACE_PATH;
    public static Context context;
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
        Assert.assertTrue(OpenCVLoader.initLocal());

        context = getTargetContext();
        Assert.assertNotNull("Context can't be 'null'", context);
        LENA_PATH = Utils.exportResource(context, context.getResources().getIdentifier("lena", "drawable", context.getPackageName()));
        CHESS_PATH = Utils.exportResource(context, context.getResources().getIdentifier("chessboard", "drawable", context.getPackageName()));
        //LBPCASCADE_FRONTALFACE_PATH = Utils.exportResource(context, R.raw.lbpcascade_frontalface);

        /*
         * The original idea about test order randomization is from
         * marek.defecinski blog.
         */
        //List<TestCase> testCases = androidTestRunner.getTestCases();
        //Collections.shuffle(testCases); //shuffle the tests order

        super.onStart();
    }

    public static String getOutputFileName(String name)
    {
        return context.getExternalFilesDir(null).getAbsolutePath() + File.separatorChar + name;
    }
}
