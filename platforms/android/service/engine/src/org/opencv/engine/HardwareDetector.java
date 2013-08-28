package org.opencv.engine;

public class HardwareDetector
{
    public static final int ARCH_UNKNOWN = -1;

    public static final int ARCH_X86      = 0x01000000;
    public static final int ARCH_X64      = 0x02000000;

    public static final int ARCH_ARMv5    = 0x04000000;
    public static final int ARCH_ARMv6    = 0x08000000;
    public static final int ARCH_ARMv7    = 0x10000000;
    public static final int ARCH_ARMv8    = 0x20000000;

    public static final int ARCH_MIPS     = 0x40000000;
    // Platform specific features
    // ! Check CPU arch before !

    // ARM specific features
    public static final int FEATURES_HAS_VFPv3d16 = 0x01;
    public static final int FEATURES_HAS_VFPv3    = 0x02;
    public static final int FEATURES_HAS_NEON     = 0x04;
    public static final int FEATURES_HAS_NEON2    = 0x08;

    // X86 specific features
    public static final int FEATURES_HAS_SSE  = 0x01;
    public static final int FEATURES_HAS_SSE2 = 0x02;
    public static final int FEATURES_HAS_SSE3 = 0x04;

    // GPU Acceleration options
    public static final int FEATURES_HAS_GPU = 0x010000;

    public static final int PLATFORM_TEGRA   = 1;
    public static final int PLATFORM_TEGRA2  = 2;
    public static final int PLATFORM_TEGRA3  = 3;
    public static final int PLATFORM_TEGRA4i = 4;
    public static final int PLATFORM_TEGRA4  = 5;
    public static final int PLATFORM_TEGRA5  = 6;

    public static final int PLATFORM_UNKNOWN = 0;

    // Return CPU arch and list of supported features
    public static native int GetCpuID();
    // Return hardware platform name
    public static native String GetPlatformName();
    // Return processor count
    public static native int GetProcessorCount();

    public static native int DetectKnownPlatforms();

    public static boolean mIsReady = false;

    static {
        try {
            System.loadLibrary("OpenCVEngine");
            System.loadLibrary("OpenCVEngine_jni");
            mIsReady = true;
        }
        catch(UnsatisfiedLinkError e) {
            mIsReady = false;
            e.printStackTrace();
        }
    }
}
