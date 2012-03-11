import sys, re, os.path
from string import Template

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

class_ignore_list = (
    #core
    "FileNode", "FileStorage", "KDTree",
    #highgui
    "VideoWriter", "VideoCapture",
    #features2d
    #"KeyPoint", "MSER", "StarDetector", "SURF", "DMatch",
)

const_ignore_list = (
    "CV_CAP_OPENNI",
    "CV_CAP_PROP_OPENNI_",
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
    "CV_CAP_PROP_MONOCROME",
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
    "CV_YUV420(i|sp)2.+",
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
    "GRIDRETECTOR",
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
            ('LINE_AA', 16), ('LINE_8', 8), ('LINE_4', 4),
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
        ) # private
    }, # Imgproc

    "Calib3d":
    {
        'private' :
        (
            ('CV_LMEDS',  4),
            ('CV_RANSAC', 8),
            ('CV_FM_LMEDS', 'CV_LMEDS'),
            ('CV_FM_RANSAC','CV_RANSAC'),
            ('CV_FM_7POINT', 1),
            ('CV_FM_8POINT', 2),
            ('CV_CALIB_USE_INTRINSIC_GUESS', 1),
            ('CV_CALIB_FIX_ASPECT_RATIO',    2),
            ('CV_CALIB_FIX_PRINCIPAL_POINT', 4),
            ('CV_CALIB_ZERO_TANGENT_DIST',   8),
            ('CV_CALIB_FIX_FOCAL_LENGTH',   16),
            ('CV_CALIB_FIX_K1',             32),
            ('CV_CALIB_FIX_K2',             64),
            ('CV_CALIB_FIX_K3',            128),
            ('CV_CALIB_FIX_K4',           2048),
            ('CV_CALIB_FIX_K5',           4096),
            ('CV_CALIB_FIX_K6',           8192),
            ('CV_CALIB_RATIONAL_MODEL',  16384),
            ('CV_CALIB_FIX_INTRINSIC',     256),
            ('CV_CALIB_SAME_FOCAL_LENGTH', 512),
            ('CV_CALIB_ZERO_DISPARITY',   1024),
        ) # public
    }, # Calib3d

}


# c_type    : { java/jni correspondence }
type_dict = {
# "simple"  : { j_type : "?", jn_type : "?", jni_type : "?", suffix : "?" },
    ""        : { "j_type" : "", "jn_type" : "long", "jni_type" : "jlong" }, # c-tor ret_type
    "void"    : { "j_type" : "void", "jn_type" : "void", "jni_type" : "void" },
    "env"     : { "j_type" : "", "jn_type" : "", "jni_type" : "JNIEnv*"},
    "cls"     : { "j_type" : "", "jn_type" : "", "jni_type" : "jclass"},
    "bool"    : { "j_type" : "boolean", "jn_type" : "boolean", "jni_type" : "jboolean", "suffix" : "Z" },
    "int"     : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "long"    : { "j_type" : "int", "jn_type" : "int", "jni_type" : "jint", "suffix" : "I" },
    "float"   : { "j_type" : "float", "jn_type" : "float", "jni_type" : "jfloat", "suffix" : "F" },
    "double"  : { "j_type" : "double", "jn_type" : "double", "jni_type" : "jdouble", "suffix" : "D" },
    "size_t"  : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "__int64" : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "int64"   : { "j_type" : "long", "jn_type" : "long", "jni_type" : "jlong", "suffix" : "J" },
    "double[]": { "j_type" : "double[]", "jn_type" : "double[]", "jni_type" : "jdoubleArray", "suffix" : "_3D" },

# "complex" : { j_type : "?", jn_args : (("", ""),), jn_name : "", jni_var : "", jni_name : "", "suffix" : "?" },

    "vector_Point"    : { "j_type" : "List<Point>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Point> %(n)s", "suffix" : "J" },
    "vector_Point2f"  : { "j_type" : "List<Point>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Point2f> %(n)s", "suffix" : "J" },
    "vector_Point2d"  : { "j_type" : "List<Point>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Point2d> %(n)s", "suffix" : "J" },
    "vector_Point3i"  : { "j_type" : "List<Point3>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Point3i> %(n)s", "suffix" : "J" },
    "vector_Point3f"  : { "j_type" : "List<Point3>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Point3f> %(n)s", "suffix" : "J" },
    "vector_Point3d"  : { "j_type" : "List<Point3>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Point3d> %(n)s", "suffix" : "J" },
    "vector_Mat"      : { "j_type" : "List<Mat>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Mat> %(n)s", "suffix" : "J" },
    "vector_KeyPoint" : { "j_type" : "List<KeyPoint>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<KeyPoint> %(n)s", "suffix" : "J" },
    "vector_DMatch"   : { "j_type" : "List<DMatch>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<DMatch> %(n)s", "suffix" : "J" },
    "vector_Rect"     : { "j_type" : "List<Rect>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Rect> %(n)s", "suffix" : "J" },
    "vector_uchar"    : { "j_type" : "List<Byte>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<uchar> %(n)s", "suffix" : "J" },
    "vector_char"     : { "j_type" : "List<Byte>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<char> %(n)s", "suffix" : "J" },
    "vector_int"      : { "j_type" : "List<Integer>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<int> %(n)s", "suffix" : "J" },
    "vector_float"    : { "j_type" : "List<Float>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<float> %(n)s", "suffix" : "J" },
    "vector_double"   : { "j_type" : "List<Double>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<double> %(n)s", "suffix" : "J" },
    "vector_Vec4f"    : { "j_type" : "Mat", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Vec4f> %(n)s", "suffix" : "J" },
    "vector_Vec6f"    : { "j_type" : "Mat", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector<Vec6f> %(n)s", "suffix" : "J" },

    "vector_vector_KeyPoint": { "j_type" : "List<List<KeyPoint>>", "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector< vector<KeyPoint> > %(n)s" },
    "vector_vector_DMatch"  : { "j_type" : "List<List<DMatch>>",   "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector< vector<DMatch> > %(n)s" },
    "vector_vector_char"    : { "j_type" : "List<List<Byte>>",     "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector< vector<char> > %(n)s" },
    "vector_vector_Point"   : { "j_type" : "List<List<Point>>",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector< vector<Point> > %(n)s" },
    "vector_vector_Point2f" : { "j_type" : "List<List<Point>>",    "jn_type" : "long", "jni_type" : "jlong", "jni_var" : "vector< vector<Point2f> > %(n)s" },

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
    "string"  : { "j_type" : "String",  "jn_type" : "String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); std::string n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
    "String"  : { "j_type" : "String",  "jn_type" : "String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); String n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
    "c_string": { "j_type" : "String",  "jn_type" : "String",
                  "jni_type" : "jstring", "jni_name" : "n_%(n)s.c_str()",
                  "jni_var" : 'const char* utf_%(n)s = env->GetStringUTFChars(%(n)s, 0); std::string n_%(n)s( utf_%(n)s ? utf_%(n)s : "" ); env->ReleaseStringUTFChars(%(n)s, utf_%(n)s)',
                  "suffix" : "Ljava_lang_String_2"},
"TermCriteria": { "j_type" : "TermCriteria",  "jn_args" : (("int", ".type"), ("int", ".maxCount"), ("double", ".epsilon")),
                  "jni_var" : "TermCriteria %(n)s(%(n)s_type, %(n)s_maxCount, %(n)s_epsilon)",
                  "suffix" : "IID"},
    "Vec3d"   : { "j_type" : "double[]",  "jn_args" : (("double", ".val[0]"), ("double", ".val[1]"), ("double", ".val[2]")),
                  "jn_type" : "double[]",
                  "jni_var" : "Vec3d %(n)s(%(n)s_val0, %(n)s_val1, %(n)s_val2)", "jni_type" : "jdoubleArray",
                  "suffix" : "DDD"},

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

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Core_n_1minMaxLocManual
  (JNIEnv* env, jclass cls, jlong src_nativeObj, jlong mask_nativeObj)
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

    } catch(cv::Exception e) {
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

        'getTextSize' :
        {
            'j_code'   :
"""
    // C++: Size getTextSize(const string& text, int fontFace, double fontScale, int thickness, int* baseLine);
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
// C++: Size getTextSize(const string& text, int fontFace, double fontScale, int thickness, int* baseLine);

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_core_Core_n_1getTextSize
  (JNIEnv* env, jclass cls, jstring text, jint fontFace, jdouble fontScale, jint thickness, jintArray baseLine)
{
    try {
        LOGD("Core::n_1getTextSize()");
        jdoubleArray result;
        result = env->NewDoubleArray(2);
        if (result == NULL) {
            return NULL; /* out of memory error thrown */
        }

        const char* utf_text = env->GetStringUTFChars(text, 0);
        std::string n_text( utf_text ? utf_text : "" );
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

        if (baseLine != NULL)
            env->SetIntArrayRegion(baseLine, 0, 1, pbaseLine);

        return result;

    } catch(cv::Exception e) {
        LOGD("Core::n_1getTextSize() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return NULL;
    } catch (...) {
        LOGD("Core::n_1getTextSize() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {core::getTextSize()}");
        return NULL;
    }
}

""",
        }, # getTextSize

        "checkHardwareSupport" : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "setUseOptimized"      : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },
        "useOptimized"         : {'j_code' : '', 'jn_code' : '', 'cpp_code' : '' },

    }, # Core

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
    }, # Highgui

}

# { class : { func : {arg_name : ctype} } }
func_arg_fix = {
    '' : {
        'randu'    : { 'low'     : 'double', 'high'   : 'double', },
        'randn'    : { 'mean'    : 'double', 'stddev' : 'double', },
        'inRange'  : { 'lowerb'  : 'Scalar', 'upperb' : 'Scalar', },
        'goodFeaturesToTrack' : { 'corners' : 'vector_Point', },
        'findFundamentalMat' : { 'points1' : 'vector_Point2d', 'points2' : 'vector_Point2d', },
        'cornerSubPix' : { 'corners' : 'vector_Point2f', },
        'minEnclosingCircle' : { 'points' : 'vector_Point2f', },
        'findHomography' : { 'srcPoints' : 'vector_Point2f', 'dstPoints' : 'vector_Point2f', },
        'solvePnP' : { 'objectPoints' : 'vector_Point3f', 'imagePoints' : 'vector_Point2f', },
        'solvePnPRansac' : { 'objectPoints' : 'vector_Point3f', 'imagePoints' : 'vector_Point2f', },
        'calcOpticalFlowPyrLK' : { 'prevPts' : 'vector_Point2f', 'nextPts' : 'vector_Point2f',
                                   'status' : 'vector_uchar', 'err' : 'vector_float', },
        'fitEllipse' : { 'points' : 'vector_Point2f', },
        'fillPoly' : { 'pts' : 'vector_vector_Point', },
        'polylines' : { 'pts' : 'vector_vector_Point', },
        'fillConvexPoly' : { 'points' : 'vector_Point', },
        'boundingRect' : { 'points' : 'vector_Point', },
        #'approxPolyDP' : { 'curve' : 'vector_Point2f', 'CV_OUT approxCurve' : 'vector_Point2f', },
        'arcLength' : { 'curve' : 'vector_Point2f', },
        'isContourConvex' : { 'contour' : 'vector_Point2f', },
        'pointPolygonTest' : { 'contour' : 'vector_Point2f', },
        'minAreaRect' : { 'points' : 'vector_Point2f', },
        'getAffineTransform' : { 'src' : 'vector_Point2f', 'dst' : 'vector_Point2f', },
        'hconcat' : { 'src' : 'vector_Mat', },
        'vconcat' : { 'src' : 'vector_Mat', },

    }, # '', i.e. no class
} # func_arg_fix

class ConstInfo(object):
    def __init__(self, cname, name, val, addedManually=False):
        self.cname = cname
        self.name = re.sub(r"^Cv", "", name)
        self.value = val
        self.addedManually = addedManually


class ClassPropInfo(object):
    def __init__(self, decl): # [f_ctype, f_name, '', '/RW']
        self.ctype = decl[0]
        self.name = decl[1]
        self.rw = "/RW" in decl[3]

class ClassInfo(object):
    def __init__(self, decl): # [ 'class/struct cname', [bases], [modlist] ]
        name = decl[0]
        name = name[name.find(" ")+1:].strip()
        self.cname = self.name = self.jname = re.sub(r"^cv\.", "", name)
        self.cname =self.cname.replace(".", "::")
        self.methods = {}
        self.methods_suffixes = {}
        self.consts = [] # using a list to save the occurence order
        self.private_consts = []
        self.imports = set()
        self.props= []
        self.jname = self.name
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]


class ArgInfo(object):
    def __init__(self, arg_tuple): # [ ctype, name, def val, [mod], argno ]
        self.pointer = False
        ctype = arg_tuple[0]
        if ctype.endswith("*"):
            ctype = ctype[:-1]
            self.pointer = True
        self.ctype = ctype
        self.name = arg_tuple[1]
        self.defval = arg_tuple[2]
        self.out = ""
        if "/O" in arg_tuple[3]:
            self.out = "O"
        if "/IO" in arg_tuple[3]:
            self.out = "IO"


class FuncInfo(object):
    def __init__(self, decl): # [ funcname, return_ctype, [modifiers], [args] ]
        name = re.sub(r"^cv\.", "", decl[0])
        self.cname = name.replace(".", "::")
        classname = ""
        dpos = name.rfind(".")
        if dpos >= 0:
            classname = name[:dpos]
            name = name[dpos+1:]
        self.classname = classname
        self.jname = self.name = name
        if "[" in name:
            self.jname = "getelem"
        for m in decl[2]:
            if m.startswith("="):
                self.jname = m[1:]
        self.static = ["","static"][ "/S" in decl[2] ]
        self.ctype = decl[1] or ""
        self.args = []
        arg_fix_map = func_arg_fix.get(classname, {}).get(self.jname, {})
        for a in decl[3]:
            arg = a[:]
            arg[0] = arg_fix_map.get(arg[1], arg[0])
            ai = ArgInfo(arg)
            self.args.append(ai)



class FuncFamilyInfo(object):
    def __init__(self, decl): # [ funcname, return_ctype, [modifiers], [args] ]
        self.funcs = []
        self.funcs.append( FuncInfo(decl) )
        self.jname = self.funcs[0].jname
        self.isconstructor = self.funcs[0].name == self.funcs[0].classname



    def add_func(self, fi):
        self.funcs.append( fi )


class JavaWrapperGenerator(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.classes = { "Mat" : ClassInfo([ 'class Mat', [], [] ]) }
        self.module = ""
        self.Module = ""
        self.java_code= {} # { class : {j_code, jn_code} }
        self.cpp_code = None
        self.ported_func_list = []
        self.skipped_func_list = []
        self.def_args_hist = {} # { def_args_cnt : funcs_cnt }
        self.classes_map = []
        self.classes_simple = []

    def add_class_code_stream(self, class_name):
        jname = self.classes[class_name].jname
        self.java_code[class_name] = { "j_code" : StringIO(), "jn_code" : StringIO(), }
        if class_name != self.Module:
            self.java_code[class_name]["j_code"].write("""
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.%(m)s;

$imports

// C++: class %(c)s
//javadoc: %(c)s
public class %(jc)s {

    protected final long nativeObj;
    protected %(jc)s(long addr) { nativeObj = addr; }

""" % { 'm' : self.module, 'c' : class_name, 'jc' : jname } )

        else: # class_name == self.Module
            self.java_code[class_name]["j_code"].write("""
//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.%(m)s;

$imports

public class %(jc)s {
""" % { 'm' : self.module, 'jc' : jname } )

        self.java_code[class_name]["jn_code"].write("""
    //
    // native stuff
    //
    static { System.loadLibrary("opencv_java"); }
""" )



    def add_class(self, decl):
        classinfo = ClassInfo(decl)
        if classinfo.name in class_ignore_list:
            return
        name = classinfo.name
        if name in self.classes:
            print "Generator error: class %s (%s) is duplicated" % \
                    (name, classinfo.cname)
            return
        self.classes[name] = classinfo
        if name in type_dict:
            print "Duplicated class: " + name
            return
        if '/Simple' in decl[2]:
            self.classes_simple.append(name)
        if ('/Map' in decl[2]) or (name == 'CvStatModel'):
            self.classes_map.append(name)
            #adding default c-tor
            ffi = FuncFamilyInfo(['cv.'+name+'.'+name, '', [], []])
            classinfo.methods[ffi.jname] = ffi
        type_dict[name] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "(*("+name+"*)%(n)s_nativeObj)", "jni_type" : "jlong",
              "suffix" : "J" }
        type_dict[name+'*'] = \
            { "j_type" : classinfo.jname,
              "jn_type" : "long", "jn_args" : (("__int64", ".nativeObj"),),
              "jni_name" : "("+name+"*)%(n)s_nativeObj", "jni_type" : "jlong",
              "suffix" : "J" }

        # missing_consts { Module : { public : [[name, val],...], private : [[]...] } }
        if name in missing_consts:
            if 'private' in missing_consts[name]:
                for (n, val) in missing_consts[name]['private']:
                    classinfo.private_consts.append( ConstInfo(n, n, val, True) )
            if 'public' in missing_consts[name]:
                for (n, val) in missing_consts[name]['public']:
                    classinfo.consts.append( ConstInfo(n, n, val, True) )

        # class props
        for p in decl[3]:
            if True: #"vector" not in p[0]:
                classinfo.props.append( ClassPropInfo(p) )
            else:
                print "Skipped property: [%s]" % name, p

        self.add_class_code_stream(name)


    def add_const(self, decl): # [ "const cname", val, [], [] ]
        name = decl[0].replace("const ", "").strip()
        name = re.sub(r"^cv\.", "", name)
        cname = name.replace(".", "::")
        for c in const_ignore_list:
            if re.match(c, name):
                return
        # class member?
        dpos = name.rfind(".")
        if dpos >= 0:
            classname = name[:dpos]
            name = name[dpos+1:]
        else:
            classname = self.Module
        if classname not in self.classes:
            # this class isn't wrapped
            # skipping this const
            return

        consts = self.classes[classname].consts
        for c in const_private_list:
            if re.match(c, name):
                consts = self.classes[classname].private_consts
                break

        constinfo = ConstInfo(cname, name, decl[1])
        # checking duplication
        for list in self.classes[classname].consts, self.classes[classname].private_consts:
            for c in list:
                if c.name == constinfo.name:
                    if c.addedManually:
                        return
                    print "Generator error: constant %s (%s) is duplicated" \
                            % (constinfo.name, constinfo.cname)
                    sys.exit(-1)

        consts.append(constinfo)

    def add_func(self, decl):
        ffi = FuncFamilyInfo(decl)
        classname = ffi.funcs[0].classname or self.Module
        if classname in class_ignore_list:
            return
        if classname in ManualFuncs and ffi.jname in ManualFuncs[classname]:
            return
        if classname not in self.classes:
            print "Generator error: the class %s for method %s is missing" % \
                    (classname, ffi.jname)
            sys.exit(-1)
        func_map = self.classes[classname].methods
        if ffi.jname in func_map:
            func_map[ffi.jname].add_func(ffi.funcs[0])
        else:
            func_map[ffi.jname] = ffi
        # calc args with def val
        cnt = len([a for a in ffi.funcs[0].args if a.defval])
        self.def_args_hist[cnt] = self.def_args_hist.get(cnt, 0) + 1

    def save(self, path, buf):
        f = open(path, "wt")
        f.write(buf)
        f.close()

    def gen(self, srcfiles, module, output_path):
        self.clear()
        self.module = module
        self.Module = module.capitalize()
        parser = hdr_parser.CppHeaderParser()

        self.add_class( ['class ' + self.Module, [], [], []] ) # [ 'class/struct cname', [bases], [modlist] [props] ]

        # scan the headers and build more descriptive maps of classes, consts, functions
        for hdr in srcfiles:
            decls = parser.parse(hdr)
            for decl in decls:
                name = decl[0]
                if name.startswith("struct") or name.startswith("class"):
                    self.add_class(decl)
                elif name.startswith("const"):
                    self.add_const(decl)
                else: # function
                    self.add_func(decl)

        #FIXME: BackgroundSubtractor is merged into BackgroundSubtractorMOG because of inheritance
        if "BackgroundSubtractor" in self.classes:
            bs = self.classes["BackgroundSubtractor"]
            bsmog = self.classes["BackgroundSubtractorMOG"]
            for name, mtd in bs.methods.items():
                for fn in mtd.funcs:
                    fn.classname = "BackgroundSubtractorMOG"
                bsmog.methods[name] = mtd;
            del self.classes["BackgroundSubtractor"]

        self.cpp_code = StringIO()
        self.cpp_code.write("""
//
// This file is auto-generated, please don't edit!
//

#include <jni.h>

#include "converters.h"

#ifdef DEBUG
#include <android/log.h>
#define MODULE_LOG_TAG "OpenCV.%(m)s"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, MODULE_LOG_TAG, __VA_ARGS__))
#else //DEBUG
#define LOGD(...)
#endif //DEBUG

#include "opencv2/%(m)s/%(m)s.hpp"

using namespace cv;

extern "C" {

""" % {'m' : module} )

        # generate code for the classes
        for name in self.classes.keys():
            if name == "Mat":
                continue
            self.gen_class(name)
            # saving code streams
            imports = "\n".join([ "import %s;" % c for c in \
                sorted(self.classes[name].imports) if not c.startswith('org.opencv.'+self.module) ])
            self.java_code[name]["j_code"].write("\n\n%s\n}\n" % self.java_code[name]["jn_code"].getvalue())
            java_code = self.java_code[name]["j_code"].getvalue()
            java_code = Template(java_code).substitute(imports = imports)
            self.save("%s/%s+%s.java" % (output_path, module, self.classes[name].jname), java_code)

        self.cpp_code.write( '\n} // extern "C"\n' )
        self.save(output_path+"/"+module+".cpp",  self.cpp_code.getvalue())

        # report
        report = StringIO()
        report.write("PORTED FUNCs LIST (%i of %i):\n\n" % \
            (len(self.ported_func_list), len(self.ported_func_list)+ len(self.skipped_func_list))
        )
        report.write("\n".join(self.ported_func_list))
        report.write("\n\nSKIPPED FUNCs LIST (%i of %i):\n\n" % \
            (len(self.skipped_func_list), len(self.ported_func_list)+ len(self.skipped_func_list))
        )
        report.write("".join(self.skipped_func_list))

        for i in self.def_args_hist.keys():
            report.write("\n%i def args - %i funcs" % (i, self.def_args_hist[i]))

        report.write("\n\nclass as MAP:\n\t" + "\n\t".join(self.classes_map))
        report.write("\n\nclass SIMPLE:\n\t" + "\n\t".join(self.classes_simple))

        self.save(output_path+"/"+module+".txt", report.getvalue())

        print "Done %i of %i funcs." % (len(self.ported_func_list), len(self.ported_func_list)+ len(self.skipped_func_list))



    def get_imports(self, scope_classname, ctype):
        imports = self.classes[scope_classname or self.Module].imports
        if ctype.startswith('vector'):
            imports.add("java.util.List")
            imports.add("org.opencv.core.Mat")
            imports.add("org.opencv.utils.Converters")
            ctype = ctype.replace('vector_', '')
        j_type = ''
        if ctype in type_dict:
            j_type = type_dict[ctype]['j_type']
        if j_type in ( "CvType", "Mat", "Point", "Point3", "Range", "Rect", "RotatedRect", "Scalar", "Size", "TermCriteria" ):
            imports.add("org.opencv.core." + j_type)
        if j_type == 'String':
            imports.add("java.lang.String")



    def gen_func(self, fi, prop_name=''):
        j_code   = self.java_code[fi.classname or self.Module]["j_code"]
        jn_code  = self.java_code[fi.classname or self.Module]["jn_code"]
        cpp_code = self.cpp_code

        # c_decl
        # e.g: void add(Mat src1, Mat src2, Mat dst, Mat mask = Mat(), int dtype = -1)
        if prop_name:
            c_decl = "%s %s::%s" % (fi.ctype, fi.classname, prop_name)
        else:
            decl_args = []
            for a in fi.args:
                s = a.ctype
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
            print "SKIP:", c_decl, "\n\tdue to RET type", fi.ctype
            return
        for a in fi.args:
            if a.ctype not in type_dict:
                msg = "// Unknown type '%s' (%s), skipping the function\n\n" % (a.ctype, a.out or "I")
                self.skipped_func_list.append(c_decl + "\n" + msg)
                j_code.write( " "*4 + msg )
                print "SKIP:", c_decl, "\n\tdue to ARG type", a.ctype, "/" + (a.out or "I")
                return

        self.ported_func_list.append(c_decl)

        # jn & cpp comment
        jn_code.write( "\n    // C++: %s\n" % c_decl )
        cpp_code.write( "\n//\n// %s\n//\n" % c_decl )

        # java args
        args = fi.args[:] # copy
        suffix_counter = int( self.classes[fi.classname or self.Module].methods_suffixes.get(fi.jname, -1) )
        while True:
            suffix_counter += 1
            self.classes[fi.classname or self.Module].methods_suffixes[fi.jname] = suffix_counter
             # java native method args
            jn_args = []
            # jni (cpp) function args
            jni_args = [ArgInfo([ "env", "env", "", [], "" ]), ArgInfo([ "cls", "cls", "", [], "" ])]
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
                    { "cnt" : len(fields), "args" : ", ".join(["_retval_" + f[1] for f in fields]) } )
            if fi.classname and fi.ctype and not fi.static: # non-static class method except c-tor
                # adding 'self'
                jn_args.append ( ArgInfo([ "__int64", "nativeObj", "", [], "" ]) )
                jni_args.append( ArgInfo([ "__int64", "self", "", [], "" ]) )
            self.get_imports(fi.classname, fi.ctype)
            for a in args:
                self.get_imports(fi.classname, a.ctype)
                if "vector" in a.ctype: # pass as Mat
                    jn_args.append  ( ArgInfo([ "__int64", "%s_mat.nativeObj" % a.name, "", [], "" ]) )
                    jni_args.append ( ArgInfo([ "__int64", "%s_mat_nativeObj" % a.name, "", [], "" ]) )
                    c_prologue.append( type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";" )
                    c_prologue.append( "Mat& %(n)s_mat = *((Mat*)%(n)s_mat_nativeObj)" % {"n" : a.name} + ";" )
                    if "I" in a.out or not a.out:
                        if a.ctype.startswith("vector_vector_"):
                            self.classes[fi.classname or self.Module].imports.add("java.util.ArrayList")
                            j_prologue.append( "List<Mat> %(n)s_tmplm = new ArrayList<Mat>((%(n)s != null) ? %(n)s.size() : 0);" % {"n" : a.name } )
                            j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s, %(n)s_tmplm);" % {"n" : a.name, "t" : a.ctype} )
                        else:
                            j_prologue.append( "Mat %(n)s_mat = Converters.%(t)s_to_Mat(%(n)s);" % {"n" : a.name, "t" : a.ctype} )
                        c_prologue.append( "Mat_to_%(t)s( %(n)s_mat, %(n)s );" % {"n" : a.name, "t" : a.ctype} )
                    else:
                        if type_dict[a.ctype]["j_type"] != "Mat":
                            j_prologue.append( "Mat %s_mat = new Mat();" % a.name )
                        else:
                            j_prologue.append( "Mat %s_mat = %s;" % (a.name, a.name) )
                    if "O" in a.out:
                        if type_dict[a.ctype]["j_type"] != "Mat":
                            j_epilogue.append("Converters.Mat_to_%(t)s(%(n)s_mat, %(n)s);" % {"t" : a.ctype, "n" : a.name})
                        c_epilogue.append( "%(t)s_to_Mat( %(n)s, %(n)s_mat );" % {"n" : a.name, "t" : a.ctype} )
                else:
                    fields = type_dict[a.ctype].get("jn_args", ((a.ctype, ""),))
                    if "I" in a.out or not a.out or a.ctype in self.classes: # input arg, pass by primitive fields
                        for f in fields:
                            jn_args.append ( ArgInfo([ f[0], a.name + f[1], "", [], "" ]) )
                            jni_args.append( ArgInfo([ f[0], a.name + f[1].replace(".","_").replace("[","").replace("]",""), "", [], "" ]) )
                    if a.out and a.ctype not in self.classes: # out arg, pass as double[]
                        jn_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        jni_args.append ( ArgInfo([ "double[]", "%s_out" % a.name, "", [], "" ]) )
                        j_prologue.append( "double[] %s_out = new double[%i];" % (a.name, len(fields)) )
                        c_epilogue.append( \
                            "jdouble tmp_%(n)s[%(cnt)i] = {%(args)s}; env->SetDoubleArrayRegion(%(n)s_out, 0, %(cnt)i, tmp_%(n)s);" %
                            { "n" : a.name, "cnt" : len(fields), "args" : ", ".join([a.name + f[1] for f in fields]) } )
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
            java_doc = "//javadoc: " + f_name + "(%s)" % ", ".join([a.name for a in args])
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
                ret_val = "Mat retValMat = new Mat("
                tail = ")"
                j_type = type_dict[ret_type]["j_type"]
                j_prologue.append( j_type + ' retVal = new Array' + j_type+'();')
                self.classes[fi.classname or self.Module].imports.add('java.util.ArrayList')
                j_epilogue.append('Converters.Mat_to_' + ret_type + '(retValMat, retVal);')
            elif ret_type == "void":
                ret_val = ""
                ret = "return;"
            elif ret_type == "": # c-tor
                ret_val = "nativeObj = "
                ret = "return;"
            elif ret_type in self.classes: # wrapped class
                ret_val = type_dict[ret_type]["j_type"] + " retVal = new " + self.classes[ret_type].jname + "("
                tail = ")"
            elif "jn_type" not in type_dict[ret_type]:
                ret_val = type_dict[fi.ctype]["j_type"] + " retVal = new " + type_dict[ret_type]["j_type"] + "("
                tail = ")"

            static = "static"
            if fi.classname:
                static = fi.static

            j_args = []
            for a in args:
                jt = type_dict[a.ctype]["j_type"]
                if a.out and a.ctype in ('bool', 'int', 'long', 'float', 'double'):
                    jt += '[]'
                j_args.append( jt + ' ' + a.name )

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
            elif fi.ctype == "string":
                ret = "return env->NewStringUTF(_retval_.c_str());"
                default = 'return env->NewStringUTF("");'
            elif fi.ctype in self.classes: # wrapped class:
                ret = "return (jlong) new %s(_retval_);" % fi.ctype
            elif ret_type in self.classes: # pointer to wrapped class:
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

            cvname = "cv::" + name
            retval = fi.ctype + " _retval_ = "
            if fi.ctype == "void":
                retval = ""
            elif fi.ctype.startswith('vector'):
                retval = type_dict[fi.ctype]['jni_var'] % {"n" : '_ret_val_vector_'} + " = "
                c_epilogue.append("Mat* _retval_ = new Mat();")
                c_epilogue.append(fi.ctype+"_to_Mat(_ret_val_vector_, *_retval_);")
            if fi.classname:
                if not fi.ctype: # c-tor
                    retval = fi.classname + "* _retval_ = "
                    cvname = "new " + fi.classname
                elif fi.static:
                    cvname = "%s::%s" % (fi.classname, name)
                else:
                    cvname = "me->" + name
                    c_prologue.append(\
                        "%(cls)s* me = (%(cls)s*) self; //TODO: check for NULL" \
                            % { "cls" : fi.classname} \
                    )
            cvargs = []
            for a in args:
                if a.pointer:
                    jni_name = "&%(n)s"
                else:
                    jni_name = "%(n)s"
                cvargs.append( type_dict[a.ctype].get("jni_name", jni_name) % {"n" : a.name})
                if "vector" not in a.ctype :
                    if ("I" in a.out or not a.out or a.ctype in self.classes) and "jni_var" in type_dict[a.ctype]: # complex type
                        c_prologue.append(type_dict[a.ctype]["jni_var"] % {"n" : a.name} + ";")
                    if a.out and "I" not in a.out and a.ctype not in self.classes:
                        c_prologue.append("%s %s;" % (a.ctype, a.name))

            rtype = type_dict[fi.ctype].get("jni_type", "jdoubleArray")
            clazz = self.Module
            if fi.classname:
                clazz = self.classes[fi.classname].jname
            cpp_code.write ( Template( \
"""

JNIEXPORT $rtype JNICALL Java_org_opencv_${module}_${clazz}_$fname
  ($args)
{
    try {
        LOGD("$module::$fname()");
        $prologue
        $retval$cvname( $cvargs );
        $epilogue
        $ret
    } catch(cv::Exception e) {
        LOGD("$module::$fname() catched cv::Exception: %s", e.what());
        jclass je = env->FindClass("org/opencv/core/CvException");
        if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        $default
    } catch (...) {
        LOGD("$module::$fname() catched unknown exception (...)");
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {$module::$fname()}");
        $default
    }
}


""" ).substitute( \
        rtype = rtype, \
        module = self.module, \
        clazz = clazz.replace('_', '_1'), \
        fname = (fi.jname + '_' + str(suffix_counter)).replace('_', '_1'), \
        args = ", ".join(["%s %s" % (type_dict[a.ctype].get("jni_type"), a.name) for a in jni_args]), \
        prologue = "\n        ".join(c_prologue), \
        epilogue = "  ".join(c_epilogue), \
        ret = ret, \
        cvname = cvname, \
        cvargs = ", ".join([a for a in cvargs]), \
        default = default, \
        retval = retval, \
    ) )

            # processing args with default values
            if args and args[-1].defval:
                a = args.pop()
            else:
                break



    def gen_class(self, name):
        # generate code for the class
        ci = self.classes[name]
        # constants
        if ci.private_consts:
            self.java_code[name]['j_code'].write("""
    private static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.private_consts])
            )
        if ci.consts:
            self.java_code[name]['j_code'].write("""
    public static final int
            %s;\n\n""" % (",\n"+" "*12).join(["%s = %s" % (c.name, c.value) for c in ci.consts])
            )
        # c-tors
        fflist = ci.methods.items()
        fflist.sort()
        for n, ffi in fflist:
            if ffi.isconstructor:
                for fi in ffi.funcs:
                    fi.jname = ci.jname
                    self.gen_func(fi)
        # other methods
        for n, ffi in fflist:
            if not ffi.isconstructor:
                for fi in ffi.funcs:
                    self.gen_func(fi)
        # props
        for pi in ci.props:
            # getter
            getter_name = name + ".get_" + pi.name
            #print getter_name
            fi = FuncInfo( [getter_name, pi.ctype, [], []] ) # [ funcname, return_ctype, [modifiers], [args] ]
            self.gen_func(fi, pi.name)
            if pi.rw:
                #setter
                setter_name = name + ".set_" + pi.name
                #print setter_name
                fi = FuncInfo( [ setter_name, "void", [], [ [pi.ctype, pi.name, "", [], ""] ] ] )
                self.gen_func(fi, pi.name)

        # manual ports
        if name in ManualFuncs:
            for func in ManualFuncs[name].keys():
                self.java_code[name]["j_code"].write ( ManualFuncs[name][func]["j_code"] )
                self.java_code[name]["jn_code"].write( ManualFuncs[name][func]["jn_code"] )
                self.cpp_code.write( ManualFuncs[name][func]["cpp_code"] )

        if name != self.Module:
            # finalize()
            self.java_code[name]["j_code"].write(
"""
    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
        super.finalize();
    }
""" )

            self.java_code[name]["jn_code"].write(
"""
    // native support for java finalize()
    private static native void delete(long nativeObj);
""" )

            # native support for java finalize()
            self.cpp_code.write( \
"""
//
//  native support for java finalize()
//  static void %(cls)s::delete( __int64 self )
//

JNIEXPORT void JNICALL Java_org_opencv_%(module)s_%(j_cls)s_delete
  (JNIEnv* env, jclass cls, jlong self)
{
    delete (%(cls)s*) self;
}

""" % {"module" : module, "cls" : name, "j_cls" : ci.jname}
            )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage:\n", \
            os.path.basename(sys.argv[0]), \
            "<full path to hdr_parser.py> <module name> <C++ header> [<C++ header>...]"
        print "Current args are: ", ", ".join(["'"+a+"'" for a in sys.argv])
        exit(0)

    dstdir = "."
    hdr_parser_path = os.path.abspath(sys.argv[1])
    if hdr_parser_path.endswith(".py"):
        hdr_parser_path = os.path.dirname(hdr_parser_path)
    sys.path.append(hdr_parser_path)
    import hdr_parser
    module = sys.argv[2]
    srcfiles = sys.argv[3:]
    print "Generating module '" + module + "' from headers:\n\t" + "\n\t".join(srcfiles)
    generator = JavaWrapperGenerator()
    generator.gen(srcfiles, module, dstdir)

