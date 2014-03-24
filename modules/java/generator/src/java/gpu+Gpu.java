package org.opencv.gpu;

public class Gpu {

    public static final int
            FEATURE_SET_COMPUTE_10 = 10,
            FEATURE_SET_COMPUTE_11 = 11,
            FEATURE_SET_COMPUTE_12 = 12,
            FEATURE_SET_COMPUTE_13 = 13,
            FEATURE_SET_COMPUTE_20 = 20,
            FEATURE_SET_COMPUTE_21 = 21,
            FEATURE_SET_COMPUTE_30 = 30,
            FEATURE_SET_COMPUTE_35 = 35,
            GLOBAL_ATOMICS = FEATURE_SET_COMPUTE_11,
            SHARED_ATOMICS = FEATURE_SET_COMPUTE_12,
            NATIVE_DOUBLE = FEATURE_SET_COMPUTE_13,
            WARP_SHUFFLE_FUNCTIONS = FEATURE_SET_COMPUTE_30,
            DYNAMIC_PARALLELISM = FEATURE_SET_COMPUTE_35;


    //
    // C++:  bool deviceSupports(int feature_set)
    //

    //javadoc: deviceSupports(feature_set)
    public static boolean deviceSupports(int feature_set)
    {
        boolean retVal = deviceSupports_0(feature_set);
        return retVal;
    }


    //
    // C++:  int getCudaEnabledDeviceCount()
    //

    //javadoc: getCudaEnabledDeviceCount()
    public static int getCudaEnabledDeviceCount()
    {
        int retVal = getCudaEnabledDeviceCount_0();
        return retVal;
    }


    //
    // C++:  int getDevice()
    //

    //javadoc: getDevice()
    public static int getDevice()
    {
        int retVal = getDevice_0();
        return retVal;
    }


    //
    // C++:  void printCudaDeviceInfo(int device)
    //

    //javadoc: printCudaDeviceInfo(device)
    public static void printCudaDeviceInfo(int device)
    {
        printCudaDeviceInfo_0(device);
        return;
    }


    //
    // C++:  void printShortCudaDeviceInfo(int device)
    //

    //javadoc: printShortCudaDeviceInfo(device)
    public static void printShortCudaDeviceInfo(int device)
    {
        printShortCudaDeviceInfo_0(device);
        return;
    }


    //
    // C++:  void resetDevice()
    //

    //javadoc: resetDevice()
    public static void resetDevice()
    {
        resetDevice_0();
        return;
    }


    //
    // C++:  void setDevice(int device)
    //

    //javadoc: setDevice(device)
    public static void setDevice(int device)
    {
        setDevice_0(device);
        return;
    }




    // C++:  bool deviceSupports(int feature_set)
    private static native boolean deviceSupports_0(int feature_set);

    // C++:  int getCudaEnabledDeviceCount()
    private static native int getCudaEnabledDeviceCount_0();

    // C++:  int getDevice()
    private static native int getDevice_0();

    // C++:  void printCudaDeviceInfo(int device)
    private static native void printCudaDeviceInfo_0(int device);

    // C++:  void printShortCudaDeviceInfo(int device)
    private static native void printShortCudaDeviceInfo_0(int device);

    // C++:  void resetDevice()
    private static native void resetDevice_0();

    // C++:  void setDevice(int device)
    private static native void setDevice_0(int device);

}
