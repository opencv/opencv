package org.opencv.core;

public final class CvType {

    // type depth constants
    public static final int
            CV_8U = 0, CV_8S = 1,
            CV_16U = 2, CV_16S = 3,
            CV_32S = 4,
            CV_32F = 5,
            CV_64F = 6,
            CV_USRTYPE1 = 7;

    // predefined type constants
    public static final int
            CV_8UC1 = CV_8UC(1), CV_8UC2 = CV_8UC(2), CV_8UC3 = CV_8UC(3), CV_8UC4 = CV_8UC(4),
            CV_8SC1 = CV_8SC(1), CV_8SC2 = CV_8SC(2), CV_8SC3 = CV_8SC(3), CV_8SC4 = CV_8SC(4),
            CV_16UC1 = CV_16UC(1), CV_16UC2 = CV_16UC(2), CV_16UC3 = CV_16UC(3), CV_16UC4 = CV_16UC(4),
            CV_16SC1 = CV_16SC(1), CV_16SC2 = CV_16SC(2), CV_16SC3 = CV_16SC(3), CV_16SC4 = CV_16SC(4),
            CV_32SC1 = CV_32SC(1), CV_32SC2 = CV_32SC(2), CV_32SC3 = CV_32SC(3), CV_32SC4 = CV_32SC(4),
            CV_32FC1 = CV_32FC(1), CV_32FC2 = CV_32FC(2), CV_32FC3 = CV_32FC(3), CV_32FC4 = CV_32FC(4),
            CV_64FC1 = CV_64FC(1), CV_64FC2 = CV_64FC(2), CV_64FC3 = CV_64FC(3), CV_64FC4 = CV_64FC(4);

    private static final int CV_DEPTH_SHIFT= 0, CV_CN_SHIFT = 5, CV_CN_LEN = 7, CV_CN_EXP_LEN = 3,
                             CV_CN_BASE_LEN = (CV_CN_LEN - CV_CN_EXP_LEN),
                             CV_DEPTH_LEN = (CV_CN_SHIFT - CV_DEPTH_SHIFT);
    private static final int CV_SANITY_DEPTH_MASK = ((1 << CV_DEPTH_LEN  ) - 1),
                             CV_SANITY_CN_EXP_MASK = ((1 << CV_CN_EXP_LEN ) - 1),
                             CV_SANITY_CN_BASE_MASK = ((1 << CV_CN_BASE_LEN) - 1),
                             CV_MAT_DEPTH_MASK = (CV_SANITY_DEPTH_MASK << CV_DEPTH_SHIFT),
                             CV_MAT_TYPE_MASK = ((1 << (CV_CN_SHIFT + CV_CN_LEN)) - 1);

    private static final int CV_DEPTH_MAX = (1 << CV_DEPTH_LEN),
                             CV_CN_MAX = (1 << (CV_CN_BASE_LEN + CV_SANITY_CN_EXP_MASK)) /*(1 << CV_CN_LEN )*/;

    private static final int channelExponent(int channels) {
        if (channels <= CV_SANITY_CN_BASE_MASK) return 0;
        return (32 - CV_CN_BASE_LEN - Integer.numberOfLeadingZeros(channels - 1)) & CV_SANITY_CN_EXP_MASK;
    }

    private static final int channelBase(int channels) {
        return (channels - 1) >> channelExponent(channels);
    }

    private static final int channelDeflate(int channels) {
        return (channelExponent(channels) << CV_CN_BASE_LEN) | ( channelBase(channels) & CV_SANITY_CN_BASE_MASK) /*channels - 1*/;
    }

    private static final int channelGetExponent(int bin_channels) {
        return (bin_channels >> CV_CN_BASE_LEN) & CV_SANITY_CN_EXP_MASK;
    }

    private static final int channelGetBase(int bin_channels) {
        return bin_channels & CV_SANITY_CN_BASE_MASK;
    }

    private static final int channelInflate(int bin_channels) {
        return (channelGetBase(bin_channels) + 1) << channelGetExponent(bin_channels) /*bin_channels + 1*/;
    }

    public static final int makeType(int depth, int channels) {
        if (channels <= 0 || channels >= CV_CN_MAX) {
            throw new java.lang.UnsupportedOperationException(
                    "Channels count should be 1.." + (CV_CN_MAX - 1));
        }
        if (depth < 0 || depth >= CV_DEPTH_MAX) {
            throw new java.lang.UnsupportedOperationException(
                    "Data type depth should be 0.." + (CV_DEPTH_MAX - 1));
        }
        return ((depth & (CV_DEPTH_MAX - 1)) << CV_DEPTH_SHIFT) + (channelDeflate(channels) << CV_CN_SHIFT);
    }

    public static final int CV_8UC(int ch) {
        return makeType(CV_8U, ch);
    }

    public static final int CV_8SC(int ch) {
        return makeType(CV_8S, ch);
    }

    public static final int CV_16UC(int ch) {
        return makeType(CV_16U, ch);
    }

    public static final int CV_16SC(int ch) {
        return makeType(CV_16S, ch);
    }

    public static final int CV_32SC(int ch) {
        return makeType(CV_32S, ch);
    }

    public static final int CV_32FC(int ch) {
        return makeType(CV_32F, ch);
    }

    public static final int CV_64FC(int ch) {
        return makeType(CV_64F, ch);
    }

    public static final int channels(int type) {
        return channelInflate((type & CV_MAT_TYPE_MASK) >> CV_CN_SHIFT);
    }

    public static final int depth(int type) {
        return (type & CV_MAT_DEPTH_MASK) >> CV_DEPTH_SHIFT;
    }

    public static final boolean isInteger(int type) {
        return depth(type) < CV_32F;
    }

    public static final int ELEM_SIZE(int type) {
        switch (depth(type)) {
        case CV_8U:
        case CV_8S:
            return channels(type);
        case CV_16U:
        case CV_16S:
            return 2 * channels(type);
        case CV_32S:
        case CV_32F:
            return 4 * channels(type);
        case CV_64F:
            return 8 * channels(type);
        default:
            throw new java.lang.UnsupportedOperationException(
                    "Unsupported CvType value: " + type);
        }
    }

    public static final String typeToString(int type) {
        String s;
        switch (depth(type)) {
        case CV_8U:
            s = "CV_8U";
            break;
        case CV_8S:
            s = "CV_8S";
            break;
        case CV_16U:
            s = "CV_16U";
            break;
        case CV_16S:
            s = "CV_16S";
            break;
        case CV_32S:
            s = "CV_32S";
            break;
        case CV_32F:
            s = "CV_32F";
            break;
        case CV_64F:
            s = "CV_64F";
            break;
        case CV_USRTYPE1:
            s = "CV_USRTYPE1";
            break;
        default:
            throw new java.lang.UnsupportedOperationException(
                    "Unsupported CvType value: " + type);
        }

        int ch = channels(type);
        if (ch <= 4)
            return s + "C" + ch;
        else
            return s + "C(" + ch + ")";
    }

}
