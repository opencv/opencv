package org.opencv.core;

//NOTE: Type constants and related functions are ported from modules/core/include/opencv2/core/hal/interface.h

public final class CvType {

    // type depth constants
    public static final int
            CV_8U = 0,
            CV_8S = 1,
            CV_16U = 2,
            CV_16S = 3,
            CV_32S = 4,
            CV_32F = 5,
            CV_64F = 6,
            CV_16F = 7,
            CV_16BF = 8,
            CV_Bool = 9,
            CV_64U = 10,
            CV_64S = 11,
            CV_32U = 12
            ;

    // predefined type constants
    public static final int
            CV_8UC1 = CV_8UC(1), CV_8UC2 = CV_8UC(2), CV_8UC3 = CV_8UC(3), CV_8UC4 = CV_8UC(4),
            CV_8SC1 = CV_8SC(1), CV_8SC2 = CV_8SC(2), CV_8SC3 = CV_8SC(3), CV_8SC4 = CV_8SC(4),
            CV_16UC1 = CV_16UC(1), CV_16UC2 = CV_16UC(2), CV_16UC3 = CV_16UC(3), CV_16UC4 = CV_16UC(4),
            CV_16SC1 = CV_16SC(1), CV_16SC2 = CV_16SC(2), CV_16SC3 = CV_16SC(3), CV_16SC4 = CV_16SC(4),
            CV_32SC1 = CV_32SC(1), CV_32SC2 = CV_32SC(2), CV_32SC3 = CV_32SC(3), CV_32SC4 = CV_32SC(4),
            CV_32UC1 = CV_32UC(1), CV_32UC2 = CV_32UC(2), CV_32UC3 = CV_32UC(3), CV_32UC4 = CV_32UC(4),
            CV_64SC1 = CV_64SC(1), CV_64SC2 = CV_64SC(2), CV_64SC3 = CV_64SC(3), CV_64SC4 = CV_64SC(4),
            CV_64UC1 = CV_64UC(1), CV_64UC2 = CV_64UC(2), CV_64UC3 = CV_64UC(3), CV_64UC4 = CV_64UC(4),
            CV_32FC1 = CV_32FC(1), CV_32FC2 = CV_32FC(2), CV_32FC3 = CV_32FC(3), CV_32FC4 = CV_32FC(4),
            CV_64FC1 = CV_64FC(1), CV_64FC2 = CV_64FC(2), CV_64FC3 = CV_64FC(3), CV_64FC4 = CV_64FC(4),
            CV_16FC1 = CV_16FC(1), CV_16FC2 = CV_16FC(2), CV_16FC3 = CV_16FC(3), CV_16FC4 = CV_16FC(4),
            CV_16BFC1 = CV_16BFC(1), CV_16BFC2 = CV_16BFC(2), CV_16BFC3 = CV_16BFC(3), CV_16BFC4 = CV_16BFC(4),
            CV_BoolC1 = CV_BoolC(1), CV_BoolC2 = CV_BoolC(2), CV_BoolC3 = CV_BoolC(3), CV_BoolC4 = CV_BoolC(4);

    private static final int CV_CN_MAX = 128, CV_CN_SHIFT = 5, CV_DEPTH_MAX = (1 << CV_CN_SHIFT);

    public static final int makeType(int depth, int channels) {
        if (channels <= 0 || channels >= CV_CN_MAX) {
            throw new UnsupportedOperationException(
                    "Channels count should be 1.." + (CV_CN_MAX - 1));
        }
        if (depth < 0 || depth >= CV_DEPTH_MAX) {
            throw new UnsupportedOperationException(
                    "Data type depth should be 0.." + (CV_DEPTH_MAX - 1));
        }
        return (depth & (CV_DEPTH_MAX - 1)) + ((channels - 1) << CV_CN_SHIFT);
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

    public static final int CV_32UC(int ch) {
        return makeType(CV_32U, ch);
    }

    public static final int CV_64SC(int ch) {
        return makeType(CV_64S, ch);
    }

    public static final int CV_64UC(int ch) {
        return makeType(CV_64U, ch);
    }

    public static final int CV_32FC(int ch) {
        return makeType(CV_32F, ch);
    }

    public static final int CV_64FC(int ch) {
        return makeType(CV_64F, ch);
    }

    public static final int CV_16FC(int ch) {
        return makeType(CV_16F, ch);
    }

    public static final int CV_16BFC(int ch) {
        return makeType(CV_16BF, ch);
    }

    public static final int CV_BoolC(int ch) {
        return makeType(CV_Bool, ch);
    }

    public static final int channels(int type) {
        return (type >> CV_CN_SHIFT) + 1;
    }

    public static final int depth(int type) {
        return type & (CV_DEPTH_MAX - 1);
    }

    public static final boolean isInteger(int type) {
        return depth(type) < CV_32F;
    }

    public static final int ELEM_SIZE(int type) {
        switch (depth(type)) {
        case CV_Bool:
        case CV_8U:
        case CV_8S:
            return channels(type);
        case CV_16U:
        case CV_16S:
        case CV_16F:
        case CV_16BF:
            return 2 * channels(type);
        case CV_32S:
        case CV_32U:
        case CV_32F:
            return 4 * channels(type);
        case CV_64U:
        case CV_64S:
        case CV_64F:
            return 8 * channels(type);
        default:
            throw new UnsupportedOperationException(
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
        case CV_32U:
            s = "CV_32U";
            break;
        case CV_32F:
            s = "CV_32F";
            break;
        case CV_64U:
            s = "CV_64U";
            break;
        case CV_64S:
            s = "CV_64S";
            break;
        case CV_64F:
            s = "CV_64F";
            break;
        case CV_16F:
            s = "CV_16F";
            break;
        case CV_16BF:
            s = "CV_16BF";
            break;
        case CV_Bool:
            s = "CV_Bool";
            break;
        default:
            throw new UnsupportedOperationException(
                    "Unsupported CvType value: " + type);
        }

        int ch = channels(type);
        if (ch <= 4)
            return s + "C" + ch;
        else
            return s + "C(" + ch + ")";
    }

}
