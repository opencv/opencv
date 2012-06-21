package org.opencv.android;

import android.content.Context;
import android.graphics.Bitmap;

import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class Utils {

    public static String exportResource(Context context, int resourceId) {
        return exportResource(context, resourceId, "OpenCV_data");
    }

    public static String exportResource(Context context, int resourceId, String dirname) {
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

    public static Mat loadResource(Context context, int resourceId) throws IOException
    {
        return loadResource(context, resourceId, -1);
    }

    public static Mat loadResource(Context context, int resourceId, int flags) throws IOException
    {
        InputStream is = context.getResources().openRawResource(resourceId);
        ByteArrayOutputStream os = new ByteArrayOutputStream(is.available());

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        is.close();

        Mat encoded = new Mat(1, os.size(), CvType.CV_8U);
        encoded.put(0, 0, os.toByteArray());
        os.close();

        Mat decoded = Highgui.imdecode(encoded, flags);
        encoded.release();

        return decoded;
    }

    /**
     * Converts Android Bitmap to OpenCV Mat.
     * <p>
     * The function converts an image in the Android Bitmap representation to the OpenCV Mat.
     * <br>The 'ARGB_8888' and 'RGB_565' input Bitmap formats are supported.
     * <br>The output Mat is always created of the same size as the input Bitmap and of the 'CV_8UC4' type,
     * it keeps the image in RGBA format.
     * <br>The function throws an exception if the conversion fails.
     *
     * @param b is a valid input Bitmap object of the type 'ARGB_8888' or 'RGB_565'.
     * @param m is a valid output Mat object, it will be reallocated if needed, so it's possible to pass an empty Mat.
     */
    public static void bitmapToMat(Bitmap b, Mat m) {
        if (b == null)
            throw new java.lang.IllegalArgumentException("Bitmap b == null");
        if (m == null)
            throw new java.lang.IllegalArgumentException("Mat m == null");
    	nBitmapToMat(b, m.nativeObj);
    }


    /**
     * Converts OpenCV Mat to Android Bitmap.
     * <p>
     * <br>The function converts an image in the OpenCV Mat representation to the Android Bitmap.
     * <br>The input Mat object has to be of the types 'CV_8UC1' (gray-scale), 'CV_8UC3' (RGB) or 'CV_8UC4' (RGBA).
     * <br>The output Bitmap object has to be of the same size as the input Mat and of the types 'ARGB_8888' or 'RGB_565'.
     * <br>The function throws an exception if the conversion fails.
     *
     * @param m is a valid input Mat object of the types 'CV_8UC1', 'CV_8UC3' or 'CV_8UC4'.
     * @param b is a valid Bitmap object of the same size as the Mat m and of type 'ARGB_8888' or 'RGB_565'.
     */
    public static void matToBitmap(Mat m, Bitmap b) {
        if (m == null)
            throw new java.lang.IllegalArgumentException("Mat m == null");
        if (b == null)
            throw new java.lang.IllegalArgumentException("Bitmap b == null");
        nMatToBitmap(m.nativeObj, b);
    }

    private static native void nBitmapToMat(Bitmap b, long m_addr);

    private static native void nMatToBitmap(long m_addr, Bitmap b);
}
