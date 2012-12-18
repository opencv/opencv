// TODO: This file is very similar to the one in android_test.

package org.opencv.test;

import java.io.File;
import java.io.IOException;
import junit.framework.Assert;

import org.opencv.core.Mat;

// import java.io.IOException;
// import java.util.ArrayList;
// import java.util.Enumeration;
// import java.util.jar.JarEntry;
// import java.util.jar.JarFile;
// import org.junit.runner.*;
// import java.io.File;

// import junit.framework.TestCase;
// import org.junit.Test;
// import org.junit.internal.TextListener;
// import org.junit.runner.JUnitCore;
//import org.junit.tests.TestSystem;

public class OpenCVTestRunner {
    public static void runSingle(String testName) {
	System.out.println(testName);

        // JUnitCore runner = new JUnitCore();
        // TestSystem system = new TestSystem();
        // Results results = system.outContents();
        // runner.addListener(new TextListener(system));

	JUnitCore junit = new JUnitCore();
	junit.addListener(new TextListener(System.out));
	//junit.run(TestReader.class);

	try {
	    Class c = Class.forName(testName);
	    junit.run(c);
	    //org.junit.runner.JUnitCore.runClasses(c);
	} catch (ClassNotFoundException e) {
	    System.out.println("failed to find class TODO");
	}
    }

    public static void run(String jarfile) {
        String[] tests = findTests(jarfile);
	runSingle(tests[0]);
	//org.junit.runner.JUnitCore.main(java.util.Arrays.copyOfRange(tests, 0, 2));

	// for (String test : tests) {
	//     runSingle(test);
	// }
	// String[] firstTests = java.util.Arrays.copyOfRange(tests, 20, 21);
	// System.out.println(firstTests[0]);
	// System.out.println("about to make core");
	// JUnitCore core = new JUnitCore();
	// System.out.println("made core");
	// try {
	//     Class c = Class.forName(firstTests[0]);
	//     Result result = core.run(c);
	//     System.out.println(result.toString());
	//     System.out.println(result.getFailureCount());
	//     System.out.println(result.getRunCount());
	//     System.out.println("here");
	// } catch (ClassNotFoundException e) {
	//     System.out.println("failed to find class TODO");
	// }
	//	org.junit.runner.JUnitCore.main(firstTests);
    }

    private static String[] findTests(String jarfile) {
        ArrayList<String> tests = new ArrayList<String>();
        try {
            JarFile jf = new JarFile(jarfile);
            for (Enumeration<JarEntry> e = jf.entries(); e.hasMoreElements();) {
                String name = e.nextElement().getName();
                if (name.endsWith(".class")
		    && !name.contains("$")
		    && !name.contains("OpenCVTest"))
                    tests.add(name.replaceAll("/", ".")
			      .substring(0, name.length() - 6));
            }
            jf.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return tests.toArray(new String[0]);
    }

    public static void main(String[] args) throws IOException {
	//	String libraryPath = System.getProperty("java.library.path");
	// TODO
	//	String newPath = "/wg/stor6_home3/echristiansen/time/2012_12/opencv/build/lib:" + libraryPath;
	//	System.setProperty("java.library.path", "/wg/stor6_home3/echristiansen/time/2012_12/opencv/build/lib");
	//System.setProperty("java.library.path", "");
	System.out.println(System.getProperty("java.library.path"));
	try {
	    System.loadLibrary("opencv_java");	
	} catch (SecurityException e) {
	    System.out.println(e.toString());
	} catch (UnsatisfiedLinkError e) {
	    System.out.println(e.toString());
	}

	Mat mat = new Mat();
	System.out.println(mat.toString());

	try {
	    cwd = new File(".").getCanonicalPath();
	} catch (IOException e) {
	    System.out.println(e);
	    return;
	}

	LENA_PATH = cwd + File.separator + "res/drawable/lena.jpg";
	CHESS_PATH = cwd + File.separator + "res/drawable/chessboard.jpg";
	LBPCASCADE_FRONTALFACE_PATH = cwd + File.separator + "res/raw/lbpcascade_frontalface.xml";

	//	System.out.println(path);
	run("build/jar/opencv-test.jar");
    }

    //    private final Class<?> clazz;

    // public OpenCVTestRunner(Class clazz) throws org.junit.runners.model.InitializationError {
    // 	System.loadLibrary("opencv_java");	
    // 	super(clazz);
    // 	//        this.clazz = clazz;
    // }
    

    //    private static final long MANAGER_TIMEOUT = 3000;
    static String cwd = "";
    public static String LENA_PATH = "";
    public static String CHESS_PATH = "";
    public static String LBPCASCADE_FRONTALFACE_PATH = "";
    //    public static Context context;

    // private AndroidTestRunner androidTestRunner;
    private static String TAG = "opencv_test_java";

    // private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(getContext()) {

    //     @Override
    //     public void onManagerConnected(int status) {
    //         switch (status) {
    //             case LoaderCallbackInterface.SUCCESS:
    //             {
    //                 Log("OpenCV loaded successfully");
    //                 synchronized (this) {
    //                     notify();
    //                 }
    //             } break;
    //             default:
    //             {
    //                 super.onManagerConnected(status);
    //             } break;
    //         }
    //     }
    // };

    public static String getTempFileName(String extension)
    {
	//        File cache = context.getCacheDir();
        if (!extension.startsWith("."))
            extension = "." + extension;
        try {
            File tmp = File.createTempFile("OpenCV", extension);
            String path = tmp.getAbsolutePath();
            tmp.delete();
            return path;
        } catch (IOException e) {
            Log("Failed to get temp file name. Exception is thrown: " + e);
        }
        return null;
    }

    static public void Log(String message) {
        System.out.println(TAG + " :: " +  message);
    }

    static public void Log(Mat m) {
        System.out.println(TAG + " :: " + m + "\n " + m.dump());
    }

    // TODO
    //    @Override
    public void onStart() {
        // // try to load internal libs
        // if (!OpenCVLoader.initDebug()) {
        //     // There is no internal OpenCV libs
        //     // Using OpenCV Manager for initialization;

        //     Log("Internal OpenCV library not found. Using OpenCV Manager for initialization");
        //     OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, getContext(), mLoaderCallback);

        //     synchronized (this) {
        //         try {
        //             wait(MANAGER_TIMEOUT);
        //         } catch (InterruptedException e) {
        //             e.printStackTrace();
        //         }
        //     }
        // } else {
        //     Log("OpenCV library found inside test package. Using it!");
        // }

        // context = getContext();
        // Assert.assertTrue("Context can't be 'null'", context != null);
	System.loadLibrary("opencv_java");

	// TODO
        // LENA_PATH = Utils.exportResource(context, R.drawable.lena);
        // CHESS_PATH = Utils.exportResource(context, R.drawable.chessboard);
        // LBPCASCADE_FRONTALFACE_PATH = Utils.exportResource(context, R.raw.lbpcascade_frontalface);

        /*
         * The original idea about test order randomization is from
         * marek.defecinski blog.
         */
        //List<TestCase> testCases = androidTestRunner.getTestCases();
        //Collections.shuffle(testCases); //shuffle the tests order

	// TODO
	//        super.onStart();
    }

    // @Override
    // protected AndroidTestRunner getAndroidTestRunner() {
    //     androidTestRunner = super.getAndroidTestRunner();
    //     return androidTestRunner;
    // }

    public static String getOutputFileName(String name)
    {
	return getTempFileName(name);
	//        return context.getExternalFilesDir(null).getAbsolutePath() + File.separatorChar + name;
    }
}
