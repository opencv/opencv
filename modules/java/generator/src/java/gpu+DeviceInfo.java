package org.opencv.gpu;

import java.lang.String;

// C++: class DeviceInfo
//javadoc: DeviceInfo
public class DeviceInfo {

    protected final long nativeObj;
    protected DeviceInfo(long addr) { nativeObj = addr; }


    //
    // C++:   DeviceInfo::DeviceInfo()
    //

    //javadoc: DeviceInfo::DeviceInfo()
    public   DeviceInfo()
    {

        nativeObj = DeviceInfo_0();

        return;
    }


    //
    // C++:   DeviceInfo::DeviceInfo(int device_id)
    //

    //javadoc: DeviceInfo::DeviceInfo(device_id)
    public   DeviceInfo(int device_id)
    {

        nativeObj = DeviceInfo_1(device_id);

        return;
    }


    //
    // C++:  int DeviceInfo::deviceID()
    //

    //javadoc: DeviceInfo::deviceID()
    public  int deviceID()
    {

        int retVal = deviceID_0(nativeObj);

        return retVal;
    }


    //
    // C++:  size_t DeviceInfo::freeMemory()
    //

    //javadoc: DeviceInfo::freeMemory()
    public  long freeMemory()
    {

        long retVal = freeMemory_0(nativeObj);

        return retVal;
    }


    //
    // C++:  bool DeviceInfo::isCompatible()
    //

    //javadoc: DeviceInfo::isCompatible()
    public  boolean isCompatible()
    {

        boolean retVal = isCompatible_0(nativeObj);

        return retVal;
    }


    //
    // C++:  int DeviceInfo::majorVersion()
    //

    //javadoc: DeviceInfo::majorVersion()
    public  int majorVersion()
    {

        int retVal = majorVersion_0(nativeObj);

        return retVal;
    }


    //
    // C++:  int DeviceInfo::minorVersion()
    //

    //javadoc: DeviceInfo::minorVersion()
    public  int minorVersion()
    {

        int retVal = minorVersion_0(nativeObj);

        return retVal;
    }


    //
    // C++:  int DeviceInfo::multiProcessorCount()
    //

    //javadoc: DeviceInfo::multiProcessorCount()
    public  int multiProcessorCount()
    {

        int retVal = multiProcessorCount_0(nativeObj);

        return retVal;
    }


    //
    // C++:  string DeviceInfo::name()
    //

    //javadoc: DeviceInfo::name()
    public  String name()
    {

        String retVal = name_0(nativeObj);

        return retVal;
    }


    //
    // C++:  void DeviceInfo::queryMemory(size_t& totalMemory, size_t& freeMemory)
    //

    //javadoc: DeviceInfo::queryMemory(totalMemory, freeMemory)
    public  void queryMemory(long totalMemory, long freeMemory)
    {
        double[] totalMemory_out = new double[1];
        double[] freeMemory_out = new double[1];
        queryMemory_0(nativeObj, totalMemory_out, freeMemory_out);
        totalMemory = (long)totalMemory_out[0];
        freeMemory = (long)freeMemory_out[0];
    }


    //
    // C++:  size_t DeviceInfo::sharedMemPerBlock()
    //

    //javadoc: DeviceInfo::sharedMemPerBlock()
    public  long sharedMemPerBlock()
    {

        long retVal = sharedMemPerBlock_0(nativeObj);

        return retVal;
    }


    //
    // C++:  bool DeviceInfo::supports(int feature_set)
    //

    //javadoc: DeviceInfo::supports(feature_set)
    public  boolean supports(int feature_set)
    {

        boolean retVal = supports_0(nativeObj, feature_set);

        return retVal;
    }


    //
    // C++:  size_t DeviceInfo::totalMemory()
    //

    //javadoc: DeviceInfo::totalMemory()
    public  long totalMemory()
    {

        long retVal = totalMemory_0(nativeObj);

        return retVal;
    }


    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }



    // C++:   DeviceInfo::DeviceInfo()
    private static native long DeviceInfo_0();

    // C++:   DeviceInfo::DeviceInfo(int device_id)
    private static native long DeviceInfo_1(int device_id);

    // C++:  int DeviceInfo::deviceID()
    private static native int deviceID_0(long nativeObj);

    // C++:  size_t DeviceInfo::freeMemory()
    private static native long freeMemory_0(long nativeObj);

    // C++:  bool DeviceInfo::isCompatible()
    private static native boolean isCompatible_0(long nativeObj);

    // C++:  int DeviceInfo::majorVersion()
    private static native int majorVersion_0(long nativeObj);

    // C++:  int DeviceInfo::minorVersion()
    private static native int minorVersion_0(long nativeObj);

    // C++:  int DeviceInfo::multiProcessorCount()
    private static native int multiProcessorCount_0(long nativeObj);

    // C++:  string DeviceInfo::name()
    private static native String name_0(long nativeObj);

    // C++:  void DeviceInfo::queryMemory(size_t& totalMemory, size_t& freeMemory)
    private static native void queryMemory_0(long nativeObj, double[] totalMemory_out, double[] freeMemory_out);

    // C++:  size_t DeviceInfo::sharedMemPerBlock()
    private static native long sharedMemPerBlock_0(long nativeObj);

    // C++:  bool DeviceInfo::supports(int feature_set)
    private static native boolean supports_0(long nativeObj, int feature_set);

    // C++:  size_t DeviceInfo::totalMemory()
    private static native long totalMemory_0(long nativeObj);

    // native support for java finalize()
    private static native void delete(long nativeObj);

}
