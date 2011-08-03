package org.opencv.android;

import org.opencv.core.CvException;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.content.Context;
import android.graphics.Bitmap;

public class Utils {

    public static String ExportResource(Context context, int resourceId) {
        return ExportResource(context, resourceId, "OpenCV_data");
    }

    public static String ExportResource(Context context, int resourceId, String dirname) {
        String fullname = context.getResources().getString(resourceId);
        String resName = fullname.substring(fullname.lastIndexOf("/") + 1);
        try {
            InputStream is = context.getResources().openRawResource(resourceId);
            File resDir = context.getDir(dirname, Context.MODE_PRIVATE);
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
            throw new CvException("Failed to export resource " + resName
                    + ". Exception thrown: " + e);
        }
    }

    public static Mat BitmapToMat(Bitmap b) {
        return new Mat(nBitmapToMat(b));
    }

    public static boolean MatToBitmap(Mat m, Bitmap b) {
        return nMatToBitmap(m.nativeObj, b);
    }

    // native stuff
    static {
        System.loadLibrary("opencv_java");
    }

    private static native long nBitmapToMat(Bitmap b);

    private static native boolean nMatToBitmap(long m, Bitmap b);
}
