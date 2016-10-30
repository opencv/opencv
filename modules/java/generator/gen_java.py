#!/usr/bin/env python

import sys, re, os.path
import logging
from pprint import pformat
from string import Template

if sys.version_info[0] >= 3:
    from io import StringIO
else:
    from cStringIO import StringIO

class_ignore_list = (
    #core
    "FileNode", "FileStorage", "KDTree", "KeyPoint", "DMatch",
    #features2d
    "SimpleBlobDetector"
)

const_ignore_list = (
    "CV_CAP_OPENNI",
    "CV_CAP_PROP_OPENNI_",
    "CV_CAP_INTELPERC",
    "CV_CAP_PROP_INTELPERC_"
    "WINDOW_AUTOSIZE",
    "CV_WND_PROP_",
    "CV_WINDOW_",
    "CV_EVENT_",
    "CV_GUI_",
    "CV_PUSH_BUTTON",
    "CV_CHECKBOX",
    "CV_RADIOBOX",

    #attention!
    #the following constants are added to this list using code automatic generation
    #TODO: should be checked
    "CV_CAP_ANY",
    "CV_CAP_MIL",
    "CV_CAP_VFW",
    "CV_CAP_V4L",
    "CV_CAP_V4L2",
    "CV_CAP_FIREWARE",
    "CV_CAP_FIREWIRE",
    "CV_CAP_IEEE1394",
    "CV_CAP_DC1394",
    "CV_CAP_CMU1394",
    "CV_CAP_STEREO",
    "CV_CAP_TYZX",
    "CV_TYZX_LEFT",
    "CV_TYZX_RIGHT",
    "CV_TYZX_COLOR",
    "CV_TYZX_Z",
    "CV_CAP_QT",
    "CV_CAP_UNICAP",
    "CV_CAP_DSHOW",
    "CV_CAP_PVAPI",
    "CV_CAP_PROP_DC1394_OFF",
    "CV_CAP_PROP_DC1394_MODE_MANUAL",
    "CV_CAP_PROP_DC1394_MODE_AUTO",
    "CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO",
    "CV_CAP_PROP_POS_MSEC",
    "CV_CAP_PROP_POS_FRAMES",
    "CV_CAP_PROP_POS_AVI_RATIO",
    "CV_CAP_PROP_FPS",
    "CV_CAP_PROP_FOURCC",
    "CV_CAP_PROP_FRAME_COUNT",
    "CV_CAP_PROP_FORMAT",
    "CV_CAP_PROP_MODE",
    "CV_CAP_PROP_BRIGHTNESS",
    "CV_CAP_PROP_CONTRAST",
    "CV_CAP_PROP_SATURATION",
    "CV_CAP_PROP_HUE",
    "CV_CAP_PROP_GAIN",
    "CV_CAP_PROP_EXPOSURE",
    "CV_CAP_PROP_CONVERT_RGB",
    "CV_CAP_PROP_WHITE_BALANCE_BLUE_U",
    "CV_CAP_PROP_RECTIFICATION",
    "CV_CAP_PROP_MONOCHROME",
    "CV_CAP_PROP_SHARPNESS",
    "CV_CAP_PROP_AUTO_EXPOSURE",
    "CV_CAP_PROP_GAMMA",
    "CV_CAP_PROP_TEMPERATURE",
    "CV_CAP_PROP_TRIGGER",
    "CV_CAP_PROP_TRIGGER_DELAY",
    "CV_CAP_PROP_WHITE_BALANCE_RED_V",
    "CV_CAP_PROP_MAX_DC1394",
    "CV_CAP_GSTREAMER_QUEUE_LENGTH",
    "CV_CAP_PROP_PVAPI_MULTICASTIP",
    "CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING",
    "EVENT_.*",
    "CV_L?(BGRA?|RGBA?|GRAY|XYZ|YCrCb|Luv|Lab|HLS|YUV|HSV)\d*2L?(BGRA?|RGBA?|GRAY|XYZ|YCrCb|Luv|Lab|HLS|YUV|HSV).*",
    "CV_COLORCVT_MAX",
    "CV_.*Bayer.*",
    "CV_YUV420(i|sp|p)2.+",
    "CV_TM_.+",
    "CV_FLOODFILL_.+",
    "CV_ADAPTIVE_THRESH_.+",
    "WINDOW_.+",
    "WND_PROP_.+",
)

const_private_list = (
    "CV_MOP_.+",
    "CV_INTER_.+",
    "CV_THRESH_.+",
    "CV_INPAINT_.+",
    "CV_RETR_.+",
    "CV_CHAIN_APPROX_.+",
    "OPPONENTEXTRACTOR",
    "GRIDDETECTOR",
    "PYRAMIDDETECTOR",
    "DYNAMICDETECTOR",
)

# { Module : { public : [[name, val],...], private : [[]...] } }
missing_consts = \
{
    'Core' :
    {
        'private' :
        (
            ('CV_8U',  0 ), ('CV_8S',  1 ),
            ('CV_16U', 2 ), ('CV_16S', 3 ),
            ('CV_32S', 4 ),
            ('CV_32F', 5 ), ('CV_64F', 6 ),
            ('CV_USRTYPE1', 7 ),
        ), # private
        'public' :
        (
            ('SVD_MODIFY_A', 1), ('SVD_NO_UV', 2), ('SVD_FULL_UV', 4),
            ('FILLED', -1),
            ('REDUCE_SUM', 0), ('REDUCE_AVG', 1), ('REDUCE_MAX', 2), ('REDUCE_MIN', 3),
        ) #public
    }, # Core

    "Imgproc":
    {
        'private' :
        (
            ('IPL_BORDER_CONSTANT',    0 ),
            ('IPL_BORDER_REPLICATE',   1 ),
            ('IPL_BORDER_REFLECT',     2 ),
            ('IPL_BORDER_WRAP',        3 ),
            ('IPL_BORDER_REFLECT_101', 4 ),
            ('IPL_BORDER_TRANSPARENT', 5 ),
        ), # private
        'public' :
        (
            ('LINE_AA', 16), ('LINE_8', 8), ('LINE_4', 4),
        ) #public
    }, # Imgproc

    "Calib3d":
    {
        'public' :
        (
            ('CALIB_USE_INTRINSIC_GUESS', '1'),
            ('CALIB_RECOMPUTE_EXTRINSIC', '2'),
            ('CALIB_CHECK_COND', '4'),
            ('CALIB_FIX_SKEW', '8'),
            ('CALIB_FIX_K1', '16'),
            ('CALIB_FIX_K2', '32'),
            ('CALIB_FIX_K3', '64'),
            ('CALIB_FIX_K4', '128'),
            ('CALIB_FIX_INTRINSIC', '256')
        )
    }, # Calib3d

    "Video":
    {
        'private' :
        (
            ('CV_LKFLOW_INITIAL_GUESSES',    4 ),
            ('CV_LKFLOW_GET_MIN_EIGENVALS',  8 ),
        ) # private
    }, # Video

}


# c_type    : { java/jni correspondence }
type_dict = {
# "simple"  : { j_type : "?", jn_type : "?", jni_type : "?", suffix : "?" },
    ""        : { "j_type" : "", "jn_type" : "long", "jni_type" : "jlong" }, # c-tor ret_type
    "void"    : { "j_type" : "void", "jn_type" : "void", "jni_type" : "void" },
    "env"     : { "j_type" : "", "jn_type" : "", "jni_type" : "JNIEnv*"},
    "cls"     : { "j_type" : "", "jn_type" : "", "jni_type" : "jclass"},
    "bool"    : { "j_type" : "boolean", "jn_type" : "boolean", "jni_type" : "jboolean", "suffix" : "Z" },
    "char"    : { "j_type" : "char", "jn_type" : "char", "jni_type" : "jchar", "suffix" : "C" },
    "int"     : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "long"    : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "float"   : { "j_type" : "float", "jn_type" : "float", "jni_type" : "jfloat", "suffix" : "F" },
    "double"  : { "j_type" : "double", "jn_type" : "double", "jni_type" : "jdouble", "suffix" : "D" },
    "size_t"  : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "__int64" : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "int64"   : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "double[]": { "j_type" : "double[]", "jn_type" : "double[]", "jni_type" : "jdoubleArray", "suffix" : "_3D" },

# "complex" : { j_type : "?", jn_args : (("", ""),), jn_name : "", jni_var : "", jni_name : "", "suffix" : "?" },

    "vector_Point"    : { "j_type" : "MatOfPoint", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Point> %(n)s", "suffix" : "J" },
    "vector_Point2f"  : { "j_type" : "MatOfPoint2f", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Point2f> %(n)s", "suffix" : "J" },
    #"vector_Point2d"  : { "j_type" : "MatOfPoint2d", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Point2d> %(n)s", "suffix" : "J" },
    "vector_Point3i"  : { "j_type" : "MatOfPoint3", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Point3i> %(n)s", "suffix" : "J" },
    "vector_Point3f"  : { "j_type" : "MatOfPoint3f", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Point3f> %(n)s", "suffix" : "J" },
    #"vector_Point3d"  : { "j_type" : "MatOfPoint3d", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Point3d> %(n)s", "suffix" : "J" },
    "vector_KeyPoint" : { "j_type" : "MatOfKeyPoint", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<KeyPoint> %(n)s", "suffix" : "J" },
    "vector_DMatch"   : { "j_type" : "MatOfDMatch", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<DMatch> %(n)s", "suffix" : "J" },
    "vector_Rect"     : { "j_type" : "MatOfRect",   "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Rect> %(n)s", "suffix" : "J" },
    "vector_uchar"    : { "j_type" : "MatOfByte",   "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<uchar> %(n)s", "suffix" : "J" },
    "vector_char"     : { "j_type" : "MatOfByte",   "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<char> %(n)s", "suffix" : "J" },
    "vector_int"      : { "j_type" : "MatOfInt",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<int> %(n)s", "suffix" : "J" },
    "vector_float"    : { "j_type" : "MatOfFloat",  "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<float> %(n)s", "suffix" : "J" },
    "vector_double"   : { "j_type" : "MatOfDouble", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<double> %(n)s", "suffix" : "J" },
    "vector_Vec4i"    : { "j_type" : "MatOfInt4",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Vec4i> %(n)s", "suffix" : "J" },
    "vector_Vec4f"    : { "j_type" : "MatOfFloat4",  "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Vec4f> %(n)s", "suffix" : "J" },
    "vector_Vec6f"    : { "j_type" : "MatOfFloat6",  "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Vec6f> %(n)s", "suffix" : "J" },

    "vector_Mat"      : { "j_type" : "List<Mat>",   "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector<Mat> %(n)s", "suffix" : "J" },

    "vector_vector_KeyPoint": { "j_type" : "List<MatOfKeyPoint>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector< std::vector<KeyPoint> > %(n)s" },
    "vector_vector_DMatch"  : { "j_type" : "List<MatOfDMatch>",   "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector< std::vector<DMatch> > %(n)s" },
    "vector_vector_char"    : { "j_type" : "List<MatOfByte>",     "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector< std::vector<char> > %(n)s" },
    "vector_vector_Point"   : { "j_type" : "List<MatOfPoint>",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector< std::vector<Point> > %(n)s" },
    "vector_vector_Point2f" : { "j_type" : "List<MatOfPoint2f>",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector< std::vector<Point2f> > %(n)s" },
    "vector_vector_Point3f" : { "j_type" : "List<MatOfPoint3f>",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "std::vector< std::vector<Point3f> > %(n)s" },

    "Mat"     : { "j_type" : "Mat", "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
                  "jni_var" : "Mat& %(n)s = *((Mat*)%(n)s_nativeObj)",
                  "jni_type" : "jlong", #"jni_name" : "*%(n)s",
                  "suffix" : "J" },

    "Point"   : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y")),
                  "jni_var" : "Point %(n)s((int)%(n)s_x, (int)%(n)s_y)", "jni_type" : "jdoubleArray",
                  "suffix" : "DD"},
    "Point2f" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y")),
                  "jni_var" : "Point2f %(n)s((float)%(n)s_x, (float)%(n)s_y)", "jni_type" : "jdoubleArray",
                  "suffix" : "DD"},
    "Point2d" : { "j_type" : "Point", "jn_args" : (("double", ".x"), ("double", ".y")),
                  "jni_var" : "Point2d %(n)s(%(n)s_x, %(n)s_y)", "jni_type" : "jdoubleArray",
                  "suffix" : "DD"},
    "Point3i" : { "j_type" : "Point3", "jn_args" : (("double", ".x"), ("double", ".y"), ("double", ".z")),
                  "jni_var" : "Point3i %(n)s((int)%(n)s_x, (int)%(n)s_y, (int)%(n)s_z)", "jni_type" : "jdoubleArray",
                  "suffix" : "DDD"},
    "Point3f" : { "j_type" : "Point3", "jn_args" : (("double", ".x"), ("double", ".y"), ("double", ".z")),
                  "jni_var" : "Point3f %(n)s((float)%(n)s_x, (float)%(n)s_y, (float)%(n)s_z)", "jni_type" : "jdoubleArray",
                  "suffix" : "DDD"},
    "Point3d" : { "j_type" : "Point3", "jn_args" : (("double", ".x"), ("double", ".y"), ("double", ".z")),
                  "jni_var" : "Point3d %(n)s(%(n)s_x, %(n)s_y, %(n)s_z)", "jni_type" : "jdoubleArray",
                  "suffix" : "DDD"},
    "KeyPoint": { "j_type" : "KeyPoint", "jn_args" : (("float", ".x"), ("float", ".y"), ("float", ".size"),
                    ("float", ".angle"), ("float", ".response"), ("int", ".octave"), ("int", ".class_id")),
                  "jni_var" : "KeyPoint %(n)s(%(n)s_x, %(n)s_y, %(n)s_size, %(n)s_angle, %(n)s_response, %(n)s_octave, %(n)s_class_id)",
                  "jni_type" : "jdoubleArray",
                  "suffix" : "FFFFFII"},
    "DMatch" :  { "j_type" : "DMatch", "jn_args" : ( ('int', 'queryIdx'), ('int', 'trainIdx'),
                    ('int', 'imgIdx'), ('float', 'distance'), ),
                  "jni_var" : "DMatch %(n)s(%(n)s_queryIdx, %(n)s_trainIdx, %(n)s_imgIdx, %(n)s_distance)",
                  "jni_type" : "jdoubleArray",
                  "suffix" : "IIIF"},
    "Rect"    : { "j_type" : "Rect",  "jn_args" : (("int", ".x"), ("int", ".y"), ("int", ".width"), ("int", ".height")),
                  "jni_var" : "Rect %(n)s(%(n)s_x, %(n)s_y, %(n)s_width, %(n)s_height)", "jni_type" : "jdoubleArray",
                  "suffix" : "IIII"},
    "Size"    : { "j_type" : "Size",  "jn_args" : (("double", ".width"), ("double", ".height")),
                  "jni_var" : "Size %(n)s((int)%(n)s_width, (int)%(n)s_height)", "jni_type" : "jdoubleArray",
                  "suffix" : "DD"},
    "Size2f"  : { "j_type" : "Size",  "jn_args" : (("double", ".width"), ("double", ".height")),
                  "jni_var" : "Size2f %(n)s((float)%(n)s_width, (float)%(n)s_height)", "jni_type" : "jdoubleArray",
                  "suffix" : "DD"},
 "RotatedRect": { "j_type" : "RotatedRect",  "jn_args" : (("double", ".center.x"), ("double", ".center.y"), ("double", ".size.width"), ("double", ".size.height"), ("double", ".angle")),
                  "jni_var" : "RotatedRect %(n)s(cv::Point2f(%(n)s_center_x, %(n)s_center_y), cv::Size2f(%(n)s_size_width, %(n)s_size_height), %(n)s_angle)",
                  "jni_type" : "jdoubleArray", "suffix" : "DDDDD"},
    "Scalar"  : { "j_type" : "Scalar",  "jn_args" : (("double", ".val[0]"), ("double", ".val[1]"), ("double", ".val[2]"), ("double", ".val[3]")),
                  "jni_var" : "Scalar %(n)s(%(n)s_val0, %(n)s_val1, %(n)s_val2, %(n)s_val3)", "jni_type" : "jdoubleArray",
                  "suffix" : "DDDD"},
    "Range"   : { "j_type" : "Range",  "jn_args" : (("int", ".start"), ("int", ".end")),
                  "jni_var" : "Range %(n)s(%(n)s_start, %(n)s_end)", "jni_type" : "jdoubleArray",
                  "suffix" : "II"},
    "CvSlice" : { "j_type" : "Range",  "jn_args" : (("int", ".start"), ("int", ".end")),
                  "jni_var" : "Range %(n)s(%(n)s_start, %(n)s_end)", "jni_type" : "jdoubleArray",
                  "suffix" : "II"},
    "String"  : { "j_type" : "String",  "jn_type" : "String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); String n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
    "c_string": { "j_type" : "String",  "jn_type" : "String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s.c_str()",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); String n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
"TermCriteria": { "j_type" : "TermCriteria",  "jn_args" : (("int", ".type"), ("int", ".maxCount"), ("double", ".epsilon")),
                  "jni_var" : "TermCriteria %(n)s(%(n)s_type, %(n)s_maxCount, %(n)s_epsilon)", "jni_type" : "jdoubleArray",
                  "suffix" : "IID"},
"CvTermCriteria": { "j_type" : "TermCriteria",  "jn_args" : (("int", ".type"), ("int", ".maxCount"), ("double", ".epsilon")),
                  "jni_var" : "TermCriteria %(n)s(%(n)s_type, %(n)s_maxCount, %(n)s_epsilon)", "jni_type" : "jdoubleArray",
                  "suffix" : "IID"},
    "Vec2d"   : { "j_type" : "double[]",  "jn_args" : (("double", ".val[0]"), ("double", ".val[1]")),
                  "jn_type" : "double[]",
                  "jni_var" : "Vec2d %(n)s(%(n)s_val0, %(n)s_val1)", "jni_type" : "jdoubleArray",
                  "suffix" : "DD"},
    "Vec3d"   : { "j_type" : "double[]",  "jn_args" : (("double", ".val[0]"), ("double", ".val[1]"), ("double", ".val[2]")),
                  "jn_type" : "double[]",
                  "jni_var" : "Vec3d %(n)s(%(n)s_val0, %(n)s_val1, %(n)s_val2)", "jni_type" : "jdoubleArray",
                  "suffix" : "DDD"},
    "Moments" : {
        "j_type" : "Moments",
        "jn_args" : (("double", ".m00"), ("double", ".m10"), ("double", ".m01"), ("double", ".m20"), ("double", ".m11"),
                     ("double", ".m02"), ("double", ".m30"), ("double", ".m21"), ("double", ".m12"), ("double", ".m03")),
        "jni_var" : "Moments %(n)s(%(n)s_m00, %(n)s_m10, %(n)s_m01, %(n)s_m20, %(n)s_m11, %(n)s_m02, %(n)s_m30, %(n)s_m21, %(n)s_m12, %(n)s_m03)",
        "jni_type" : "jdoubleArray",
        "suffix" : "DDDDDDDDDD"},

}

# { class : { func : {j_code, jn_code, cpp_code} } }
ManualFuncs = {
    'Core' :
    {
        'minMaxLoc' : {
            'j_code'   : """
    // manual port
    public static class MinMaxLocResult {
        public double minVal;
        public double maxVal;
        public Point minLoc;
        public Point maxLoc;

        public MinMaxLocResult() {
            minVal=0; maxVal=0;
            minLoc=new Point();
            maxLoc=new Point();
        }
    }

    // C++: minMaxLoc(Mat src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())

    //javadoc: minMaxLoc(src, mask)
    public static MinMaxLocResult minMaxLoc(Mat src, Mat mask) {
        MinMaxLocResult res = new MinMaxLocResult();
        long maskNativeObj=0;
        if (mask != null) {
            maskNativeObj=mask.nativeObj;
        }
        double resarr[] = n_minMaxLocManual(src.nativeObj, maskNativeObj);
        res.minVal=resarr[0];
        res.maxVal=resarr[1];
        res.minLoc.x=resarr[2];
        res.minLoc.y=resarr[3];
        res.maxLoc.x=resarr[4];
        res.maxLoc.y=resarr[5];
        return res;
    }

    //javadoc: minMaxLoc(src)
    public static MinMaxLocResult minMaxLoc(Mat src) {
        return minMaxLoc(src, null);
    }

""",
            'jn_code'  :
"""    private static native double[] n_minMaxLocManual(long src_nativeObj, long mask_nativeObj);\n""",
            'cpp_code' :
"""
// C++: minMaxLoc(Mat src, double* minVal, double* maxVal=0, Point* minLoc=0, Point* maxLoc=0, InputArray mask=noArray())
JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Core_n_1minMaxLocManual (JNIEnv*, jclass, jlong, jlong);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Core_n_1minMaxLocManual
  (JNIEnv* env, jclass, jlong src_nativeObj, jlong mask_nativeObj)
{
    try {
        LOGD("Core::n_1minMaxLoc()");
        jdoubleArray result;
        result = env->NewDoubleArray(6);
        if (result == NULL) {
            return NULL; /* out of memory error thrown */
        }

        Mat& src = *((Mat*)src_nativeObj);

        double minVal, maxVal;
        Point minLoc, maxLoc;
        if (mask_nativeObj != 0) {
            Mat& mask = *((Mat*)mask_nativeObj);
            minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, mask);
        } else {
            minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
        }

        jdouble fill[6];
        fill[0]=minVal;
        fill[1]=maxVal;
        fill[2]=minLoc.x;
        fill[3]=minLoc.y;
        fill[4]=maxLoc.x;
        fill[5]=maxLoc.y;

        env->SetDoubleArrayRegion(result, 0, 6, fill);

    return result;

    } catch(const cv::Exception& e) {
        LOGD("Core::n_1minMaxLoc() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return NULL;
    } catch (...) {
        LOGD("Core::n_1minMaxLoc() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {core::minMaxLoc()}");
        return NULL;
    }
}

""",
        }, # minMaxLoc


##        "checkRange"           : #TBD
##            {'j_code' : '/* TBD: checkRange() */', 'jn_code' : '', 'cpp_code' : '' },

        "checkHardwareSupport" : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "setUseOptimized"      : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "useOptimized"         : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },

    }, # Core

    'Imgproc' :
    {
        'getTextSize' :
        {
            'j_code'   :
    """
    // C++: Size getTextSize(const String& text, int fontFace, double fontScale, int thickness, int* baseLine);
    //javadoc:getTextSize(text, fontFace, fontScale, thickness, baseLine)
    public static Size getTextSize(String text, int fontFace, double fontScale, int thickness, int[] baseLine) {
        if(baseLine != null && baseLine.length != 1)
            throw new java.lang.IllegalArgumentException("'baseLine' must be 'int[1]' or 'null'.");
        Size retVal = new Size(n_getTextSize(text, fontFace, fontScale, thickness, baseLine));
        return retVal;
    }
    """,
            'jn_code'  :
    """    private static native double[] n_getTextSize(String text, int fontFace, double fontScale, int thickness, int[] baseLine);\n""",
            'cpp_code' :
    """
    // C++: Size getTextSize(const String& text, int fontFace, double fontScale, int thickness, int* baseLine);
    JNIEXPORT jdoubleArray JNICALL Java_org_opencv_imgproc_Imgproc_n_1getTextSize (JNIEnv*, jclass, jstring, jint, jdouble, jint, jintArray);

    JNIEXPORT jdoubleArray JNICALL Java_org_opencv_imgproc_Imgproc_n_1getTextSize
    (JNIEnv* env, jclass, jstring text, jint fontFace, jdouble fontScale, jint thickness, jintArray baseLine)
    {
    try {
        LOGD("Core::n_1getTextSize()");
        jdoubleArray result;
        result = env->NewDoubleArray(2);
        if (result == NULL) {
            return NULL; /* out of memory error thrown */
        }

        const char* utf_text = env->GetStringUTFChars(text, 0);
        String n_text( utf_text ? utf_text : "" );
        env->ReleaseStringUTFChars(text, utf_text);

        int _baseLine;
        int* pbaseLine = 0;

        if (baseLine != NULL)
            pbaseLine = &_baseLine;

        cv::Size rsize = cv::getTextSize(n_text, (int)fontFace, (double)fontScale, (int)thickness, pbaseLine);

        jdouble fill[2];
        fill[0]=rsize.width;
        fill[1]=rsize.height;

        env->SetDoubleArrayRegion(result, 0, 2, fill);

        if (baseLine != NULL) {
            jint jbaseLine = (jint)(*pbaseLine);
            env->SetIntArrayRegion(baseLine, 0, 1, &jbaseLine);
        }

        return result;

    } catch(const cv::Exception& e) {
        LOGD("Imgproc::n_1getTextSize() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return NULL;
    } catch (...) {
        LOGD("Imgproc::n_1getTextSize() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {core::getTextSize()}");
        return NULL;
    }
    }
    """,
        }, # getTextSize

    }, # Imgproc

    'Highgui' :
    {
        "namedWindow"       : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "destroyWindow"     : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "destroyAllWindows" : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "startWindowThread" : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "setWindowProperty" : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "getWindowProperty" : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "getTrackbarPos"    : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "setTrackbarPos"    : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "imshow"            : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "waitKey"           : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "moveWindow"        : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "resizeWindow"      : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
    }, # Highgui
}

# { class : { func : { arg_name : {"ctype" : ctype, "attrib" : [attrib]} } } }
func_arg_fix = {
    '' : {
        'randu'    : { 'low'  : {"ctype" : 'double'},
                       'high' : {"ctype"    : 'double'} },
        'randn'    : { 'mean'   : {"ctype" : 'double'},
                       'stddev' : {"ctype"  : 'double'} },
        'inRange'  : { 'lowerb' : {"ctype" : 'Scalar'},
                       'upperb' : {"ctype" : 'Scalar'} },
        'goodFeaturesToTrack' : { 'corners' : {"ctype" : 'vector_Point'} },
        'findFundamentalMat'  : { 'points1' : {"ctype" : 'vector_Point2f'},
                                  'points2' : {"ctype" : 'vector_Point2f'} },
        'cornerSubPix' : { 'corners' : {"ctype" : 'vector_Point2f'} },
        'minEnclosingCircle' : { 'points' : {"ctype" : 'vector_Point2f'} },
        'findHomography' : { 'srcPoints' : {"ctype" : 'vector_Point2f'},
                             'dstPoints' : {"ctype" : 'vector_Point2f'} },
        'solvePnP' : { 'objectPoints' : {"ctype" : 'vector_Point3f'},
                      'imagePoints'   : {"ctype" : 'vector_Point2f'},
                      'distCoeffs'    : {"ctype" : 'vector_double' } },
        'solvePnPRansac' : { 'objectPoints' : {"ctype" : 'vector_Point3f'},
                             'imagePoints'  : {"ctype" : 'vector_Point2f'},
                             'distCoeffs'   : {"ctype" : 'vector_double' } },
        'calcOpticalFlowPyrLK' : { 'prevPts' : {"ctype" : 'vector_Point2f'},
                                   'nextPts' : {"ctype" : 'vector_Point2f'},
                                   'status'  : {"ctype" : 'vector_uchar'},
                                   'err'     : {"ctype" : 'vector_float'} },
        'fitEllipse' : { 'points' : {"ctype" : 'vector_Point2f'} },
        'fillPoly'   : { 'pts' : {"ctype" : 'vector_vector_Point'} },
        'polylines'  : { 'pts' : {"ctype" : 'vector_vector_Point'} },
        'fillConvexPoly' : { 'points' : {"ctype" : 'vector_Point'} },
        'boundingRect'   : { 'points' : {"ctype" : 'vector_Point'} },
        'approxPolyDP' : { 'curve'       : {"ctype" : 'vector_Point2f'},
                           'approxCurve' : {"ctype" : 'vector_Point2f'} },
        'arcLength' : { 'curve' : {"ctype" : 'vector_Point2f'} },
        'pointPolygonTest' : { 'contour' : {"ctype" : 'vector_Point2f'} },
        'minAreaRect' : { 'points' : {"ctype" : 'vector_Point2f'} },
        'getAffineTransform' : { 'src' : {"ctype" : 'vector_Point2f'},
                                 'dst' : {"ctype" : 'vector_Point2f'} },
        'hconcat' : { 'src' : {"ctype" : 'vector_Mat'} },
        'vconcat' : { 'src' : {"ctype" : 'vector_Mat'} },
        'undistortPoints' : { 'src' : {"ctype" : 'vector_Point2f'},
                              'dst' : {"ctype" : 'vector_Point2f'} },
        'checkRange' : {'pos' : {"ctype" : '*'} },
        'meanStdDev' : { 'mean'   : {"ctype" : 'vector_double'},
                         'stddev' : {"ctype" : 'vector_double'} },
        'drawContours' : {'contours' : {"ctype" : 'vector_vector_Point'} },
        'findContours' : {'contours' : {"ctype" : 'vector_vector_Point'} },
        'convexityDefects' : { 'contour'          : {"ctype" : 'vector_Point'},
                               'convexhull'       : {"ctype" : 'vector_int'},
                               'convexityDefects' : {"ctype" : 'vector_Vec4i'} },
        'isContourConvex' : { 'contour' : {"ctype" : 'vector_Point'} },
        'convexHull' : { 'points' : {"ctype" : 'vector_Point'},
                         'hull'   : {"ctype" : 'vector_int'},
                         'returnPoints' : {"ctype" : ''} },
        'projectPoints' : { 'objectPoints' : {"ctype" : 'vector_Point3f'},
                            'imagePoints'  : {"ctype" : 'vector_Point2f'},
                            'distCoeffs'   : {"ctype" : 'vector_double' } },
        'initCameraMatrix2D' : { 'objectPoints' : {"ctype" : 'vector_vector_Point3f'},
                                 'imagePoints'  : {"ctype" : 'vector_vector_Point2f'} },
        'findChessboardCorners' : { 'corners' : {"ctype" : 'vector_Point2f'} },
        'drawChessboardCorners' : { 'corners' : {"ctype" : 'vector_Point2f'} },
        'mixChannels' : { 'dst' : {"attrib" : []} },
    }, # '', i.e. no class
} # func_arg_fix

def getLibVersion(version_hpp_path):
    version_file = open(version_hpp_path, "rt").read()
    major = re.search("^W*#\W*define\W+CV_VERSION_MAJOR\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    minor = re.search("^W*#\W*define\W+CV_VERSION_MINOR\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    revision = re.search("^W*#\W*define\W+CV_VERSION_REVISION\W+(\d+)\W*$", version_file, re.MULTILINE).group(1)
    status = re.search("^W*#\W*define\W+CV_VERSION_STATUS\W+\"(.*?)\"\W*$", version_file, re.MULTILINE).group(1)
    return (major, minor, revision, status)

def libVersionBlock():
    (major, minor, revision, status) = getLibVersion(
    (os.path.dirname(__file__) or '.') + '/../../core/include/opencv2/core/version.hpp')
    version_str    = '.'.join( (major, minor, revision) ) + status
    version_suffix =  ''.join( (major, minor, revision) )
    return """
    // these constants are wrapped inside functions to prevent inlining
    private static String getVersion() { return "%(v)s"; }
    private static String getNativeLibraryName() { return "opencv_java%(vs)s"; }
    private static int getVersionMajor() { return %(ma)s; }
    private static int getVersionMinor() { return %(mi)s; }
    private static int getVersionRevision() { return %(re)s; }
    private static String getVersionStatus() { return "%(st)s"; }

    public static final String VERSION = getVersion();
    public static final String NATIVE_LIBRARY_NAME = getNativeLibraryName();
    public static final int VERSION_MAJOR = getVersionMajor();
    public static final int VERSION_MINOR = getVersionMinor();
    public static final int VERSION_REVISION = getVersionRevision();
    public static final String VERSION_STATUS = getVersionStatus();
""" % { 'v' : version_str, 'vs' : version_suffix, 'ma' : major, 'mi' : minor, 're' : revision, 'st': status }


T_JAVA_START_INHERITED = """
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.$module;

$imports

// C++: class $name
//javadoc: $name
public class $jname extends $base {

    protected $jname(long addr) { super(addr); }

"""

T_JAVA_START_ORPHAN = """
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.$module;

$imports

// C++: class $name
//javadoc: $name
public class $jname {

    protected final long nativeObj;
    protected $jname(long addr) { nativeObj = addr; }

"""

T_JAVA_START_MODULE = """
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.$module;

$imports

public class $jname {
"""

T_CPP_MODULE = """
//
// This file is auto-generated, please don't edit!
//

#define LOG_TAG "org.opencv.$m"

#include "common.h"

#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_$M

#include <string>

#include "opencv2/$m.hpp"

$includes

using namespace cv;

/// throw java exception
static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method) {
  std::string what = "unknown exception";
  jclass je = 0;

  if(e) {
    std::string exception_type = "std::exception";

    if(dynamic_cast<const cv::Exception*>(e)) {
      exception_type = "cv::Exception";
      je = env->FindClass("org/opencv/core/CvException");
    }

    what = exception_type + ": " + e->what();
  }

  if(!je) je = env->FindClass("java/lang/Exception");
  env->ThrowNew(je, what.c_str());

  LOGE("%s caught %s", method, what.c_str());
  (void)method;        // avoid "unused" warning
}


extern "C" {

$code

} // extern "C"

#endif // HAVE_OPENCV_$M
"""

class GeneralInfo():
    def __init__(self, name, namespaces):
        self.namespace, self.classpath, self.classname, self.name = self.parseName(name, namespaces)

    def parseName(self, name, namespaces):
        '''
        input: full name and available namespaces
        returns: (namespace, classpath, classname, name)
        '''
        name = name[name.find(" ")+1:].strip() # remove struct/class/const prefix
        spaceName = ""
        localName = name # <classes>.<name>
        for namespace in sorted(namespaces, key=len, reverse=True):
            if name.startswith(namespace + "."):
                spaceName = namespace
                localName = name.replace(namespace + ".", "")
                break
        pieces = localName.split(".")
        if len(pieces) > 2: # <class>.<class>.<class>.<name>
            return spaceName, ".".join(pieces[:-1]), pieces[-2], pieces[-1]
        elif len(pieces) == 2: # <class>.<name>
            return spaceName, pieces[0], pieces[0], pieces[1]
        elif len(pieces) == 1: # <name>
            return spaceName, "", "", pieces[0]
        else:
            return spaceName, "", "" # error?!

    def fullName(self, isCPP=False):
        result = ".".join([self.fullClass(), self.name])
        return result if not isCPP else result.replace(".", "::")

    def fullClass(self, isCPP=False):
        result = ".".join([f for f in [self.namespace] + self.classpath.split(".") if len(f)>0])
        return result if not isCPP else result.replace(".", "::")

class ConstInfo(GeneralInfo):
    def __init__(self, decl, addedManually=False, namespaces=[]):
        GeneralInfo.__init__(self, decl[0], namespaces)
        self.cname = self.name.replace(".", "::")
        self.value = decl[1]
        self.addedManually = addedManually

    def __repr__(self):
        return Template("CONST $name=$value$manual").substitute(name=self.name,
                                                                 value=self.value,
                                                                 manual="(manual)" if self.addedManually else "")

    def isIgnored(self):
        for c in const_ignore_list:
            if re.match(c, self.name):
                return True
        return False

class ClassPropInfo():
    def __init__(self, decl): # [f_ctype, f_name, '', '/RW']
        self.ctype = decl[0]
        self.name = decl[1]
        self.rw = "/RW" in decl[3]

    def __repr__(self):
        return Template("PROP $ctype $name").substitute(ctype=self.ctype, name=self.name)

class ClassInfo(GeneralInfo):
    def __init__(self, decl, namespaces=[]): # [ 'class/struct cname', ': base', [modlist] ]
        GeneralInfo.__init__(self, decl[0], namespaces)
        self.cname = self.name.replace(".", "::")
        self.methods = []
        self.methods_suffixes = {}
        self.consts = [] # using a list to save the occurence order
        self.private_consts = []
        self.imports = set()
        self.props= []
        self.jname = self.name
        self.smart = None # True if class stores Ptr<T>* instead of T* in nativeObj field
        self.j_code = None # java code stream
        self.jn_code = None # jni code stream
        self.cpp_code = None # cpp code stream
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.base = ''
        if decl[1]:
            #self.base = re.sub(r"\b"+self.jname+r"\b", "", decl[1].replace(":", "")).strip()
            self.base = re.sub(r"^.*:", "", decl[1].split(",")[0]).strip().replace(self.jname, "")

    def __repr__(self):
        return Template("CLASS $namespace::$classpath.$name : $base").substitute(**self.__dict__)

    def getAllImports(self, module):
        return ["import %s;" % c for c in sorted(self.imports) if not c.startswith('org.opencv.'+module)]

    def addImports(self, ctype):
        if ctype.startswith('vector_vector'):
            self.imports.add("org.opencv.core.Mat")
            self.imports.add("org.opencv.utils.Converters")
            self.imports.add("java.util.List")
            self.imports.add("java.util.ArrayList")
            self.addImports(ctype.replace('vector_vector', 'vector'))
        elif ctype.startswith('Feature2D'):
            self.imports.add("org.opencv.features2d.Feature2D")
        elif ctype.startswith('vector'):
            self.imports.add("org.opencv.core.Mat")
            self.imports.add('java.util.ArrayList')
            if type_dict[ctype]['j_type'].startswith('MatOf'):
                self.imports.add("org.opencv.core." + type_dict[ctype]['j_type'])
            else:
                self.imports.add("java.util.List")
                self.imports.add("org.opencv.utils.Converters")
                self.addImports(ctype.replace('vector_', ''))
        else:
            j_type = ''
            if ctype in type_dict:
                j_type = type_dict[ctype]['j_type']
            elif ctype in ("Algorithm"):
                j_type = ctype
            if j_type in ( "CvType", "Mat", "Point", "Point3", "Range", "Rect", "RotatedRect", "Scalar", "Size", "TermCriteria", "Algorithm" ):
                self.imports.add("org.opencv.core." + j_type)
            if j_type == 'String':
                self.imports.add("java.lang.String")

    def getAllMethods(self):
        result = []
        result.extend([fi for fi in sorted(self.methods) if fi.isconstructor])
        result.extend([fi for fi in sorted(self.methods) if not fi.isconstructor])
        return result

    def addMethod(self, fi):
        self.methods.append(fi)

    def getConst(self, name):
        for cand in self.consts + self.private_consts:
            if cand.name == name:
                return cand
        return None

    def addConst(self, constinfo):
        # choose right list (public or private)
        consts = self.consts
        for c in const_private_list:
            if re.match(c, constinfo.name):
                consts = self.private_consts
                break
        consts.append(constinfo)

    def initCodeStreams(self, Module):
        self.j_code = StringIO()
        self.jn_code = StringIO()
        self.cpp_code = StringIO();
        if self.name != Module:
            self.j_code.write(T_JAVA_START_INHERITED if self.base else T_JAVA_START_ORPHAN)
        else:
            self.j_code.write(T_JAVA_START_MODULE)
        # misc handling
        if self.name == 'Core':
            self.imports.add("java.lang.String")
            self.j_code.write(libVersionBlock())

    def cleanupCodeStreams(self):
        self.j_code.close()
        self.jn_code.close()
        self.cpp_code.close()

    def generateJavaCode(self, m, M):
        return Template(self.j_code.getvalue() + "\n\n" + \
                         self.jn_code.getvalue() + "\n}\n").substitute(\
                            module = m,
                            name = self.name,
                            jname = self.jname,
                            imports = "\n".join(self.getAllImports(M)),
                            base = self.base)

    def generateCppCode(self):
        return self.cpp_code.getvalue()

class ArgInfo():
    def __init__(self, arg_tuple): # [ ctype, name, def val, [mod], argno ]
        self.pointer = False
        ctype = arg_tuple[0]
        if ctype.endswith("*"):
            ctype = ctype[:-1]
            self.pointer = True
        if ctype == 'vector_Point2d':
            ctype = 'vector_Point2f'
        elif ctype == 'vector_Point3d':
            ctype = 'vector_Point3f'
        self.ctype = ctype
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.out = ""
        if "/O" in arg_tuple[3]:
            self.out = "O"
        if "/IO" in arg_tuple[3]:
            self.out = "IO"

    def __repr__(self):
        return Template("ARG $ctype$p $name=$defval").substitute(ctype=self.ctype,
                                                                  p=" *" if self.pointer else "",
                                                                  name=self.name,
                                                                  defval=self.defval)

class FuncInfo(GeneralInfo):
    def __init__(self, decl, namespaces=[]): # [ funcname, return_ctype, [modifiers], [args] ]
        GeneralInfo.__init__(self, decl[0], namespaces)
        self.cname = self.name.replace(".", "::")
        self.jname = self.name
        self.isconstructor = self.name == self.classname
        if "[" in self.name:
            self.jname = "getelem"
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = re.sub(r"^CvTermCriteria", "TermCriteria", decl[1] or "")
        self.args = []
        func_fix_map = func_arg_fix.get(self.classname, {}).get(self.jname, {})
        for a in decl[3]:
            arg = a[:]
            arg_fix_map = func_fix_map.get(arg[1], {})
            arg[0] = arg_fix_map.get('ctype',  arg[0]) #fixing arg type
            arg[3] = arg_fix_map.get('attrib', arg[3]) #fixing arg attrib
            self.args.append(ArgInfo(arg))

    def __repr__(self):
        return Template("FUNC <$ctype $namespace.$classpath.$name $args>").substitute(**self.__dict__)

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()


class JavaWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.namespaces = set(["cv"])
        self.classes = { "Mat" : ClassInfo([ 'class Mat', '', [], [] ], self.namespaces) }
        self.module = ""
        self.Module = ""
        self.ported_func_list = []
        self.skipped_func_list = []
        self.def_args_hist = {} # { def_args_cnt : funcs_cnt }

    def add_class(self, decl):
        classinfo = ClassInfo(decl, namespaces=self.namespaces)
        if classinfo.name in class_ignore_list:
            logging.info('ignored: %s', classinfo)
            return
        name = classinfo.name
        if self.isWrapped(name):
            logging.warning('duplicated: %s', classinfo)
            return
        self.classes[name] = classinfo
        if name in type_dict:
            logging.warning('duplicated: %s', classinfo)
            return
        type_dict[name] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "(*("+classinfo.fullName(isCPP=True)+"*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J" }
        type_dict[name+'*'] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "("+classinfo.fullName(isCPP=True)+"*)%(n)s_nativeObj", "jni_type" : "jlong",
              "suffix" : "J" }

        # missing_consts { Module : { public : [[name, val],...], private : [[]...] } }
        if name in missing_consts:
            if 'private' in missing_consts[name]:
                for (n, val) in missing_consts[name]['private']:
                    classinfo.private_consts.append( ConstInfo([n, val], addedManually=True) )
            if 'public' in missing_consts[name]:
                for (n, val) in missing_consts[name]['public']:
                    classinfo.consts.append( ConstInfo([n, val], addedManually=True) )

        # class props
        for p in decl[3]:
            if True: #"vector" not in p[0]:
                classinfo.props.append( ClassPropInfo(p) )
            else:
                logging.warning("Skipped property: [%s]" % name, p)

        if classinfo.base:
            classinfo.addImports(classinfo.base)
        type_dict["Ptr_"+name] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "Ptr<"+classinfo.fullName(isCPP=True)+">(("+classinfo.fullName(isCPP=True)+"*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J" }
        logging.info('ok: class %s, name: %s, base: %s', classinfo, name, classinfo.base)

    def add_const(self, decl): # [ "const cname", val, [], [] ]
        constinfo = ConstInfo(decl, namespaces=self.namespaces)
        if constinfo.isIgnored():
            logging.info('ignored: %s', constinfo)
        elif not self.isWrapped(constinfo.classname):
            logging.info('class not found: %s', constinfo)
        else:
            ci = self.getClass(constinfo.classname)
            duplicate = ci.getConst(constinfo.name)
            if duplicate:
                if duplicate.addedManually:
                    logging.info('manual: %s', constinfo)
                else:
                    logging.warning('duplicated: %s', constinfo)
            else:
                ci.addConst(constinfo)
                logging.info('ok: %s', constinfo)

    def add_func(self, decl):
        fi = FuncInfo(decl, namespaces=self.namespaces)
        classname = fi.classname or self.Module
        if classname in class_ignore_list:
            logging.info('ignored: %s', fi)
        elif classname in ManualFuncs and fi.jname in ManualFuncs[classname]:
            logging.info('manual: %s', fi)
        elif not self.isWrapped(classname):
            logging.warning('not found: %s', fi)
        else:
            self.getClass(classname).addMethod(fi)
            logging.info('ok: %s', fi)
            # calc args with def val
            cnt = len([a for a in fi.args if a.defval])
            self.def_args_hist[cnt] = self.def_args_hist.get(cnt, 0) + 1

    def save(self, path, buf):
        f = open(path, "wt")
        f.write(buf)
        f.close()

    def gen(self, srcfiles, module, output_path, common_headers):
        self.clear()
        self.module = module
        self.Module = module.capitalize()
        # TODO: support UMat versions of declarations (implement UMat-wrapper for Java)
        parser = hdr_parser.CppHeaderParser(generate_umat_decls=False)

        self.add_class( ['class ' + self.Module, '', [], []] ) # [ 'class/struct cname', ':bases', [modlist] [props] ]

        # scan the headers and build more descriptive maps of classes, consts, functions
        includes = [];
        for hdr in common_headers:
            logging.info("\n===== Common header : %s =====", hdr)
            includes.append('#include "' + hdr + '"')
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            self.namespaces = parser.namespaces
            logging.info("\n\n===== Header: %s =====", hdr)
            logging.info("Namespaces: %s", parser.namespaces)
            if decls:
                includes.append('#include "' + hdr + '"')
            else:
                logging.info("Ignore header: %s", hdr)
            for decl in decls:
                logging.info("\n--- Incoming ---\n%s", pformat(decl, 4))
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_class(decl)
                elif name.startswith("const"):
                    self.add_const(decl)
                else: # function
                    self.add_func(decl)

        logging.info("\n\n===== Generating... =====")
        moduleCppCode = StringIO()
        for ci in self.classes.values():
            if ci.name == "Mat":
                continue
            ci.initCodeStreams(self.Module)
            self.gen_class(ci)
            classJavaCode = ci.generateJavaCode(self.module, self.Module)
            self.save("%s/%s+%s.java" % (output_path, module, ci.jname), classJavaCode)
            moduleCppCode.write(ci.generateCppCode())
            ci.cleanupCodeStreams()
        self.save(output_path+"/"+module+".cpp", Template(T_CPP_MODULE).substitute(m = module, M = module.upper(), code = moduleCppCode.getvalue(), includes = "\n".join(includes)))
        self.save(output_path+"/"+module+".txt", self.makeReport())

    def makeReport(self):
        '''
        Returns string with generator report
        '''
        report = StringIO()
        total_count = len(self.ported_func_list)+ len(self.skipped_func_list)
        report.write("PORTED FUNCs LIST (%i of %i):\n\n" % (len(self.ported_func_list), total_count))
        report.write("\n".join(self.ported_func_list))
        report.write("\n\nSKIPPED FUNCs LIST (%i of %i):\n\n" % (len(self.skipped_func_list), total_count))
        report.write("".join(self.skipped_func_list))
        for i in self.def_args_hist.keys():
            report.write("\n%i def args - %i funcs" % (i, self.def_args_hist[i]))
        return report.getvalue()

    def fullTypeName(self, t):
        if self.isWrapped(t):
            return self.getClass(t).fullName(isCPP=True)
        else:
            return t

    def gen_func(self, ci, fi, prop_name=''):
        logging.info("%s", fi)
        j_code   = ci.j_code
        jn_code  = ci.jn_code
        cpp_code = ci.cpp_code

        # c_decl
        # e.g: void add(Mat src1, Mat src2, Mat dst, Mat mask = Mat(), int dtype = -1)
        if prop_name:
            c_decl = "%s %s::%s" % (fi.ctype, fi.classname, prop_name)
        else:
            decl_args = []
            for a in fi.args:
                s = a.ctype or ' _hidden_ '
                if a.pointer:
                    s += "*"
                elif a.out:
                    s += "&"
                s += " " + a.name
                if a.defval:
                    s += " = "+a.defval
                decl_args.append(s)
            c_decl = "%s %s %s(%s)" % ( fi.static, fi.ctype, fi.cname, ", ".join(decl_args) )

        # java comment
        j_code.write( "\n    //\n    // C++: %s\n    //\n\n" % c_decl )
        # check if we 'know' all the types
        if fi.ctype not in type_dict: # unsupported ret type
            msg = "// Return type '%s' is not supported, skipping the function\n\n" % fi.ctype
            self.skipped_func_list.append(c_decl + "\n" + msg)
            j_code.write( " "*4 + msg )
            logging.warning("SKIP:" + c_decl.strip() + "\t due to RET type" + fi.ctype)
            return
        for a in fi.args:
            if a.ctype not in type_dict:
                if not a.defval and a.ctype.endswith("*"):
                    a.defval = 0
                if a.defval:
                    a.ctype = ''
                    continue
                msg = "// Unknown type '%s' (%s), skipping the function\n\n" % (a.ctype, a.out or "I")
                self.skipped_func_list.append(c_decl + "\n" + msg)
                j_code.write( " "*4 + msg )
                logging.warning("SKIP:" + c_decl.strip() + "\t due to ARG type" + a.ctype + "/" + (a.out or "I"))
                return

        self.ported_func_list.append(c_decl)

        # jn & cpp comment
        jn_code.write( "\n    // C++: %s\n" % c_decl )
        cpp_code.write( "\n//\n// %s\n//\n" % c_decl )

        # java args
        args = fi.args[:] # copy
        j_signatures=[]
        suffix_counter = int(ci.methods_suffixes.get(fi.jname, -1))
        while True:
            suffix_counter += 1
            ci.methods_suffixes[fi.jname] = suffix_counter
             # java native method args
            jn_args = []
            # jni (cpp) function args
            jni_args = [ArgInfo([ "env", "env", "", [], "" ]), ArgInfo([ "cls", "", "", [], "" ])]
            j_prologue = []
            j_epilogue = []
            c_prologue = []
            c_epilogue = []
            if type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                fields = type_dict[fi.ctype]["jn_args"]
                c_epilogue.append( \
                    ("jdoubleArray _da_retval_ = env->NewDoubleArray(%(cnt)i);  " +
                     "jdouble _tmp_retval_[%(cnt)i] = {%(args)s}; " +
                     "env->SetDoubleArrayRegion(_da_retval_, 0, %(cnt)i, _tmp_retval_);") %
                    { "cnt" : len(fields), "args" : ", ".join(["(jdouble)_retval_" + f[1] for f in fields]) } )
            if fi.classname and fi.ctype and not fi.static: # non-static class method except c-tor
                # adding 'self'
                jn_args.append ( ArgInfo([ "__int64", "nativeObj", "", [], "" ]) )
                jni_args.append( ArgInfo([ "__int64", "self", "", [], "" ]) )
            ci.addImports(fi.ctype)
            for a in args:
                if not a.ctype: # hidden
                    continue
                ci.addImports(a.ctype)
                if "vector" in a.ctype: # pass as Mat
                    jn_args.append  ( ArgInfo([ "__int64", "%s_mat.nativeObj" % a.name, "", [], "" ]) )
                    jni_args.append ( ArgInfo([ "__int64", "%s_mat_nativeObj" % a.name, "", [], "" ]) )
                    c_prologue.append( type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";" )
                    c_prologue.append( "Mat& %(n)s_mat = *((Mat*)%(n)s_mat_nativeObj)" % {"n" : a.name} + ";" )
                    if "I" in a.out or not a.out:
                        if a.ctype.startswith("vector_vector_"):
                            j_prologue.append( "List<Mat> %(n)s_tmplm = new ArrayList<Mat>((%(n)s != null) ? %(n)s.size() : 0);" % {"n" : a.name } )
                            j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s, %(n)s_tmplm);" % {"n" : a.name, "t" : a.ctype} )
                        else:
                            if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                                j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s);" % {"n" : a.name, "t" : a.ctype} )
                            else:
                                j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                        c_prologue.append( "Mat_to_%(t)s( %(n)s_mat, %(n)s );" % {"n" : a.name, "t" : a.ctype} )
                    else:
                        if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                            j_prologue.append( "Mat %s_mat = new Mat();" % a.name )
                        else:
                            j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                    if "O" in a.out:
                        if not type_dict[a.ctype]["j_type"].startswith("MatOf"):
                            j_epilogue.append("Converters.Mat_to_%(t)s(%(n)s_mat, %(n)s);" % {"t" : a.ctype, "n" : a.name})
                            j_epilogue.append( "%s_mat.release();" % a.name )
                        c_epilogue.append( "%(t)s_to_Mat( %(n)s, %(n)s_mat );" % {"n" : a.name, "t" : a.ctype} )
                else:
                    fields = type_dict[a.ctype].get("jn_args", ((a.ctype, ""),))
                    if "I" in a.out or not a.out or self.isWrapped(a.ctype): # input arg, pass by primitive fields
                        for f in fields:
                            jn_args.append ( ArgInfo([ f[0], a.name + f[1], "", [], "" ]) )
                            jni_args.append( ArgInfo([ f[0], a.name + f[1].replace(".","_").replace("[","").replace("]",""), "", [], "" ]) )
                    if a.out and not self.isWrapped(a.ctype): # out arg, pass as double[]
                        jn_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        j_prologue.append( "double[] %s_out = new double[%i];" % (a.name, len(fields)) )
                        c_epilogue.append( \
                            "jdouble tmp_%(n)s[%(cnt)i] = {%(args)s}; env->SetDoubleArrayRegion(%(n)s_out, 0, %(cnt)i, tmp_%(n)s);" %
                            { "n" : a.name, "cnt" : len(fields), "args" : ", ".join(["(jdouble)" + a.name + f[1] for f in fields]) } )
                        if a.ctype in ('bool', 'int', 'long', 'float', 'double'):
                            j_epilogue.append('if(%(n)s!=null) %(n)s[0] = (%(t)s)%(n)s_out[0];' % {'n':a.name,'t':a.ctype})
                        else:
                            set_vals = []
                            i = 0
                            for f in fields:
                                set_vals.append( "%(n)s%(f)s = %(t)s%(n)s_out[%(i)i]" %
                                    {"n" : a.name, "t": ("("+type_dict[f[0]]["j_type"]+")", "")[f[0]=="double"], "f" : f[1], "i" : i}
                                )
                                i += 1
                            j_epilogue.append( "if("+a.name+"!=null){ " + "; ".join(set_vals) + "; } ")

            # calculate java method signature to check for uniqueness
            j_args = []
            for a in args:
                if not a.ctype: #hidden
                    continue
                jt = type_dict[a.ctype]["j_type"]
                if a.out and a.ctype in ('bool', 'int', 'long', 'float', 'double'):
                    jt += '[]'
                j_args.append( jt + ' ' + a.name )
            j_signature = type_dict[fi.ctype]["j_type"] + " " + \
                fi.jname + "(" + ", ".join(j_args) + ")"
            logging.info("java: " + j_signature)

            if(j_signature in j_signatures):
                if args:
                    pop(args)
                    continue
                else:
                    break

            # java part:
            # private java NATIVE method decl
            # e.g.
            # private static native void add_0(long src1, long src2, long dst, long mask, int dtype);
            jn_code.write( Template(\
                "    private static native $type $name($args);\n").substitute(\
                type = type_dict[fi.ctype].get("jn_type", "double[]"), \
                name = fi.jname + '_' + str(suffix_counter), \
                args = ", ".join(["%s %s" % (type_dict[a.ctype]["jn_type"], a.name.replace(".","_").replace("[","").replace("]","")) for a in jn_args])
            ) );

            # java part:

            #java doc comment
            f_name = fi.name
            if fi.classname:
                f_name = fi.classname + "::" + fi.name
            java_doc = "//javadoc: " + f_name + "(%s)" % ", ".join([a.name for a in args if a.ctype])
            j_code.write(" "*4 + java_doc + "\n")

            # public java wrapper method impl (calling native one above)
            # e.g.
            # public static void add( Mat src1, Mat src2, Mat dst, Mat mask, int dtype )
            # { add_0( src1.nativeObj, src2.nativeObj, dst.nativeObj, mask.nativeObj, dtype );  }
            ret_type = fi.ctype
            if fi.ctype.endswith('*'):
                ret_type = ret_type[:-1]
            ret_val = type_dict[ret_type]["j_type"] + " retVal = "
            tail = ""
            ret = "return retVal;"
            if ret_type.startswith('vector'):
                tail = ")"
                j_type = type_dict[ret_type]["j_type"]
                if j_type.startswith('MatOf'):
                    ret_val += j_type + ".fromNativeAddr("
                else:
                    ret_val = "Mat retValMat = new Mat("
                    j_prologue.append( j_type + ' retVal = new Array' + j_type+'();')
                    j_epilogue.append('Converters.Mat_to_' + ret_type + '(retValMat, retVal);')
            elif ret_type.startswith("Ptr_"):
                ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + type_dict[ret_type]["j_type"] + "("
                tail = ")"
            elif ret_type == "void":
                ret_val = ""
                ret = "return;"
            elif ret_type == "": # c-tor
                if fi.classname and ci.base:
                    ret_val = "super( "
                    tail = " )"
                else:
                    ret_val = "nativeObj = "
                ret = "return;"
            elif self.isWrapped(ret_type): # wrapped class
                ret_val = type_dict[ret_type]["j_type"] + " retVal = new " + self.getClass(ret_type).jname + "("
                tail = ")"
            elif "jn_type" not in type_dict[ret_type]:
                ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + type_dict[ret_type]["j_type"] + "("
                tail = ")"

            static = "static"
            if fi.classname:
                static = fi.static

            j_code.write( Template(\
"""    public $static $j_type $j_name($j_args)
    {
        $prologue
        $ret_val$jn_name($jn_args_call)$tail;
        $epilogue
        $ret
    }

"""
                ).substitute(\
                    ret = ret, \
                    ret_val = ret_val, \
                    tail = tail, \
                    prologue = "\n        ".join(j_prologue), \
                    epilogue = "\n        ".join(j_epilogue), \
                    static=static, \
                    j_type=type_dict[fi.ctype]["j_type"], \
                    j_name=fi.jname, \
                    j_args=", ".join(j_args), \
                    jn_name=fi.jname + '_' + str(suffix_counter), \
                    jn_args_call=", ".join( [a.name for a in jn_args] ),\
                )
            )


            # cpp part:
            # jni_func(..) { _retval_ = cv_func(..); return _retval_; }
            ret = "return _retval_;"
            default = "return 0;"
            if fi.ctype == "void":
                ret = "return;"
                default = "return;"
            elif not fi.ctype: # c-tor
                ret = "return (jlong) _retval_;"
            elif fi.ctype.startswith('vector'): # c-tor
                ret = "return (jlong) _retval_;"
            elif fi.ctype == "String":
                ret = "return env->NewStringUTF(_retval_.c_str());"
                default = 'return env->NewStringUTF("");'
            elif self.isWrapped(fi.ctype): # wrapped class:
                ret = "return (jlong) new %s(_retval_);" % self.fullTypeName(fi.ctype)
            elif fi.ctype.startswith('Ptr_'):
                c_prologue.append("typedef Ptr<%s> %s;" % (self.fullTypeName(fi.ctype[4:]), fi.ctype))
                ret = "return (jlong)(new %(ctype)s(_retval_));" % { 'ctype':fi.ctype }
            elif self.isWrapped(ret_type): # pointer to wrapped class:
                ret = "return (jlong) _retval_;"
            elif type_dict[fi.ctype]["jni_type"] == "jdoubleArray":
                ret = "return _da_retval_;"

            # hack: replacing func call with property set/get
            name = fi.name
            if prop_name:
                if args:
                    name = prop_name + " = "
                else:
                    name = prop_name + ";//"

            cvname = fi.fullName(isCPP=True)
            retval = self.fullTypeName(fi.ctype) + " _retval_ = "
            if fi.ctype == "void":
                retval = ""
            elif fi.ctype == "String":
                retval = "cv::" + retval
            elif fi.ctype.startswith('vector'):
                retval = type_dict[fi.ctype]['jni_var'] % {"n" : '_ret_val_vector_'} + " = "
                c_epilogue.append("Mat* _retval_ = new Mat();")
                c_epilogue.append(fi.ctype+"_to_Mat(_ret_val_vector_, *_retval_);")
            if len(fi.classname)>0:
                if not fi.ctype: # c-tor
                    retval = fi.fullClass(isCPP=True) + "* _retval_ = "
                    cvname = "new " + fi.fullClass(isCPP=True)
                elif fi.static:
                    cvname = fi.fullName(isCPP=True)
                else:
                    cvname = ("me->" if  not self.isSmartClass(ci) else "(*me)->") + name
                    c_prologue.append(\
                        "%(cls)s* me = (%(cls)s*) self; //TODO: check for NULL" \
                            % { "cls" : self.smartWrap(ci, fi.fullClass(isCPP=True))} \
                    )
            cvargs = []
            for a in args:
                if a.pointer:
                    jni_name = "&%(n)s"
                else:
                    jni_name = "%(n)s"
                    if not a.out and not "jni_var" in type_dict[a.ctype]:
                        # explicit cast to C type to avoid ambiguous call error on platforms (mingw)
                        # where jni types are different from native types (e.g. jint is not the same as int)
                        jni_name  = "(%s)%s" % (a.ctype, jni_name)
                if not a.ctype: # hidden
                    jni_name = a.defval
                cvargs.append( type_dict[a.ctype].get("jni_name", jni_name) % {"n" : a.name})
                if "vector" not in a.ctype :
                    if ("I" in a.out or not a.out or self.isWrapped(a.ctype)) and "jni_var" in type_dict[a.ctype]: # complex type
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                    if a.out and "I" not in a.out and not self.isWrapped(a.ctype) and a.ctype:
                        c_prologue.append("%s %s;" % (a.ctype, a.name))

            rtype = type_dict[fi.ctype].get("jni_type", "jdoubleArray")
            clazz = ci.jname
            cpp_code.write ( Template( \
"""
${namespace}

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname ($argst);

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname
  ($args)
{
    static const char method_name[] = "$module::$fname()";
    try {
        LOGD("%s", method_name);
        $prologue
        $retval$cvname( $cvargs );
        $epilogue$ret
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }
    $default
}


""" ).substitute( \
        rtype = rtype, \
        module = self.module.replace('_', '_1'), \
        clazz = clazz.replace('_', '_1'), \
        fname = (fi.jname + '_' + str(suffix_counter)).replace('_', '_1'), \
        args  = ", ".join(["%s %s" % (type_dict[a.ctype].get("jni_type"), a.name) for a in jni_args]), \
        argst = ", ".join([type_dict[a.ctype].get("jni_type") for a in jni_args]), \
        prologue = "\n        ".join(c_prologue), \
        epilogue = "  ".join(c_epilogue) + ("\n        " if c_epilogue else ""), \
        ret = ret, \
        cvname = cvname, \
        cvargs = ", ".join(cvargs), \
        default = default, \
        retval = retval, \
        namespace = ('using namespace ' + ci.namespace.replace('.', '::') + ';') if ci.namespace else ''
    ) )

            # adding method signature to dictionarry
            j_signatures.append(j_signature)

            # processing args with default values
            if not args or not args[-1].defval:
                break
            while args and args[-1].defval:
                # 'smart' overloads filtering
                a = args.pop()
                if a.name in ('mask', 'dtype', 'ddepth', 'lineType', 'borderType', 'borderMode', 'criteria'):
                    break



    def gen_class(self, ci):
        logging.info("%s", ci)
        # constants
        if ci.private_consts:
            logging.info("%s", ci.private_consts)
            ci.j_code.write("""
    private static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.private_consts])
            )
        if ci.consts:
            logging.info("%s", ci.consts)
            ci.j_code.write("""
    public static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.consts])
            )
        # methods
        for fi in ci.getAllMethods():
            self.gen_func(ci, fi)
        # props
        for pi in ci.props:
            # getter
            getter_name = ci.fullName() + ".get_" + pi.name
            fi = FuncInfo( [getter_name, pi.ctype, [], []], self.namespaces ) # [ funcname, return_ctype, [modifiers], [args] ]
            self.gen_func(ci, fi, pi.name)
            if pi.rw:
                #setter
                setter_name = ci.fullName() + ".set_" + pi.name
                fi = FuncInfo( [ setter_name, "void", [], [ [pi.ctype, pi.name, "", [], ""] ] ], self.namespaces)
                self.gen_func(ci, fi, pi.name)

        # manual ports
        if ci.name in ManualFuncs:
            for func in ManualFuncs[ci.name].keys():
                ci.j_code.write ( ManualFuncs[ci.name][func]["j_code"] )
                ci.jn_code.write( ManualFuncs[ci.name][func]["jn_code"] )
                ci.cpp_code.write( ManualFuncs[ci.name][func]["cpp_code"] )

        if ci.name != self.Module:
            # finalize()
            ci.j_code.write(
"""
    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }
""" )

            ci.jn_code.write(
"""
    // native support for java finalize()
    private static native void delete(long nativeObj);
""" )

            # native support for java finalize()
            ci.cpp_code.write( \
"""
//
//  native support for java finalize()
//  static void %(cls)s::delete( __int64 self )
//
JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete(JNIEnv*, jclass, jlong);

JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete
  (JNIEnv*, jclass, jlong self)
{
    delete (%(cls)s*) self;
}

""" % {"module" : module.replace('_', '_1'), "cls" : self.smartWrap(ci, ci.fullName(isCPP=True)), "j_cls" : ci.jname.replace('_', '_1')}
            )

    def getClass(self, classname):
        return self.classes[classname or self.Module]

    def isWrapped(self, classname):
        name = classname or self.Module
        return name in self.classes

    def isSmartClass(self, ci):
        '''
        Check if class stores Ptr<T>* instead of T* in nativeObj field
        '''
        if ci.smart != None:
            return ci.smart

        # if parents are smart (we hope) then children are!
        # if not we believe the class is smart if it has "create" method
        ci.smart = False
        if ci.base:
            ci.smart = True
        else:
            for fi in ci.methods:
                if fi.name == "create":
                    ci.smart = True
                    break

        return ci.smart

    def smartWrap(self, ci, fullname):
        '''
        Wraps fullname with Ptr<> if needed
        '''
        if self.isSmartClass(ci):
            return "Ptr<" + fullname + ">"
        return fullname


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:\n", \
            os.path.basename(sys.argv[0]), \
            "<full path to hdr_parser.py> <module name> <C++ header> [<C++ header>...]")
        print("Current args are: ", ", ".join(["'"+a+"'" for a in sys.argv]))
        exit(0)

    dstdir = "."
    hdr_parser_path = os.path.abspath(sys.argv[1])
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser
    module = sys.argv[2]
    srcfiles = sys.argv[3:]
    common_headers = []
    if '--common' in srcfiles:
        pos = srcfiles.index('--common')
        common_headers = srcfiles[pos+1:]
        srcfiles = srcfiles[:pos]
    logging.basicConfig(filename='%s/%s.log' % (dstdir, module), format=None, filemode='w', level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    #print("Generating module '" + module + "' from headers:\n\t" + "\n\t".join(srcfiles))
    generator = JavaWrapperGenerator()
    generator.gen(srcfiles, module, dstdir, common_headers)
