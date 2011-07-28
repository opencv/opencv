package org.opencv.test;

import android.content.Context;
import android.test.AndroidTestRunner;
import android.test.InstrumentationTestRunner;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

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

    private AndroidTestRunner androidTestRunner;
    private static String TAG = "opencv_test_java";

    static public void Log(String message) {
        Log.e(TAG, message);
    }
    
    @Override
    public void onStart() {
        LENA_PATH = ExportResource(R.drawable.lena);
        CHESS_PATH = ExportResource(R.drawable.chessboard);
        LBPCASCADE_FRONTALFACE_PATH = ExportResource(R.raw.lbpcascade_frontalface);

        // List<TestCase> testCases = androidTestRunner.getTestCases();
        // Collections.shuffle(testCases); //shuffle the tests order

        super.onStart();
    }

    @Override
    protected AndroidTestRunner getAndroidTestRunner() {
        androidTestRunner = super.getAndroidTestRunner();
        return androidTestRunner;
    }

    private String ExportResource(int resourceId) {
        String fullname = getContext().getResources().getString(resourceId);
        String resName = fullname.substring(fullname.lastIndexOf("/") + 1);
        try {
            InputStream is = getContext().getResources().openRawResource(
                    resourceId);
            File resDir = getContext().getDir("testdata", Context.MODE_PRIVATE);
            File resFile = new File(resDir, resName);

            FileOutputStream os = new FileOutputStream(resFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            return resFile.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
            Log("Failed to export resource " + resName + ". Exception thrown: "
                    + e);
        }
        return null;
    }
}
